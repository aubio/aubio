/* 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public License
 * as published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *  
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307, USA
 */

/* This file orginally taken from, Fluidsynth, Peter Hanappe */

#include "aubio_priv.h"
#include "midi.h"
#include "midi_event.h"
#include "midi_parser.h"

/** aubio_midi_parser_t */
struct _aubio_midi_parser_t {
  unsigned char status;       /**< Identifies the type of event, that is 
                                 currently received ('Noteon', 'Pitch Bend'
                                 etc). */
  unsigned char channel;      /**< The channel of the event that is received 
                                (in case of a channel event) */
  unsigned int nr_bytes;      /**< How many bytes have been read for the 
                                 current event? */
  unsigned int nr_bytes_total;/**< How many bytes does the current event 
                                type include? */
  unsigned short p[AUBIO_MIDI_PARSER_MAX_PAR]; /**< The parameters */

  aubio_midi_event_t event;   /**< The event, that is returned to the 
                                 MIDI driver. */
};

/** new_aubio_midi_parser */
aubio_midi_parser_t* new_aubio_midi_parser()
{
  aubio_midi_parser_t* parser;
  parser = AUBIO_NEW(aubio_midi_parser_t);
  if (parser == NULL) {
    AUBIO_ERR("Out of memory");
    return NULL;
  }
  /* As long as the status is 0, the parser won't do anything -> no need to
   * initialize all the fields. */
  parser->status = 0; 
  return parser;
}

/** del_aubio_midi_parser */
int del_aubio_midi_parser(aubio_midi_parser_t* parser)
{
  AUBIO_FREE(parser);
  return AUBIO_OK;
}

/** aubio_midi_parser_parse
 *
 * The MIDI byte stream is fed into the parser, one byte at a time.
 * As soon as the parser has recognized an event, it will return it.
 * Otherwise it returns NULL.
 */
aubio_midi_event_t* aubio_midi_parser_parse(aubio_midi_parser_t* parser, 
    unsigned char c)
{
  /*********************************************************************/
  /* 'Process' system real-time messages                               */
  /*********************************************************************/
  /* There are not too many real-time messages that are of interest here.
   * They can occur anywhere, even in the middle of a noteon message! 
   * Real-time range: 0xF8 .. 0xFF
   * Note: Real-time does not affect (running) status.
   */  
  if (c >= 0xF8){
    if (c == MIDI_SYSTEM_RESET){
      parser->event.type = c;
      parser->status = 0; /* clear the status */
      return &parser->event;
    };
    return NULL;
  };

  /*********************************************************************/
  /* 'Process' system common messages (again, just skip them)          */
  /*********************************************************************/
  /* There are no system common messages that are of interest here.
   * System common range: 0xF0 .. 0xF7 
   */

  if (c > 0xF0){
    /* MIDI specs say: To ignore a non-real-time message, just discard all data
     * up to the next status byte.  And our parser will ignore data that is
     * received without a valid status.  
     * Note: system common cancels running status. */
    parser->status = 0;
    return NULL;
  };

  /*********************************************************************/
  /* Process voice category messages:                                  */
  /*********************************************************************/
  /* Now that we have handled realtime and system common messages, only
   * voice messages are left.
   * Only a status byte has bit # 7 set.
   * So no matter the status of the parser (in case we have lost sync),
   * as soon as a byte >= 0x80 comes in, we are dealing with a status byte
   * and start a new event.
   */

  if (c & 0x80){
    parser->channel = c & 0x0F;
    parser->status = c & 0xF0;
    /* The event consumes x bytes of data... (subtract 1 for the status 
     * byte) */
    parser->nr_bytes_total=aubio_midi_event_length(parser->status)-1;
    /* of which we have read 0 at this time. */
    parser->nr_bytes = 0;
    return NULL;
  };

  /*********************************************************************/
  /* Process data                                                      */
  /*********************************************************************/
  /* If we made it this far, then the received char belongs to the data
   * of the last event. */
  if (parser->status == 0){
    /* We are not interested in the event currently received.
     * Discard the data. */
    return NULL;
  };

  /* Store the first couple of bytes */
  if (parser->nr_bytes < AUBIO_MIDI_PARSER_MAX_PAR){
    parser->p[parser->nr_bytes]=c;
  };
  parser->nr_bytes++;

  /* Do we still need more data to get this event complete? */
  if (parser->nr_bytes < parser->nr_bytes_total){
    return NULL;
  };

  /*********************************************************************/
  /* Send the event                                                    */
  /*********************************************************************/
  /* The event is ready-to-go.  About 'running status': 
   * 
   * The MIDI protocol has a built-in compression mechanism. If several similar
   * events are sent in-a-row, for example note-ons, then the event type is
   * only sent once. For this case, the last event type (status) is remembered.
   * We simply keep the status as it is, just reset the parameter counter. If
   * another status byte comes in, it will overwrite the status. 
   */
  parser->event.type = parser->status;
  parser->event.channel = parser->channel;
  parser->nr_bytes = 0; /* Related to running status! */
  switch (parser->status){
    case NOTE_OFF:
    case NOTE_ON:
    case KEY_PRESSURE:
    case CONTROL_CHANGE:
    case PROGRAM_CHANGE:
    case CHANNEL_PRESSURE:
      parser->event.param1 = parser->p[0]; /* For example key number */
      parser->event.param2 = parser->p[1]; /* For example velocity */
      break;
    case PITCH_BEND:
      /* Pitch-bend is transmitted with 14-bit precision. */
      /* Note: '|' does here the same as '+' (no common bits), 
       * but might be faster */
      parser->event.param1 = ((parser->p[1] << 7) | parser->p[0]); 
      break;
    default: 
      /* Unlikely */
      return NULL;
  };
  return &parser->event;
};



/* Taken from Nagano Daisuke's USB-MIDI driver */
static int remains_f0f6[] = {
  0,	/* 0xF0 */
  2,	/* 0XF1 */
  3,	/* 0XF2 */
  2,	/* 0XF3 */
  2,	/* 0XF4 (Undefined by MIDI Spec, and subject to change) */
  2,	/* 0XF5 (Undefined by MIDI Spec, and subject to change) */
  1	  /* 0XF6 */
};

static int remains_80e0[] = {
  3,	/* 0x8X Note Off */
  3,	/* 0x9X Note On */
  3,	/* 0xAX Poly-key pressure */
  3,	/* 0xBX Control Change */
  2,	/* 0xCX Program Change */
  2,	/* 0xDX Channel pressure */
  3 	/* 0xEX PitchBend Change */
};

/** Returns the length of the MIDI message starting with c.
 *
 * Taken from Nagano Daisuke's USB-MIDI driver */
int aubio_midi_event_length(unsigned char event){
  if ( event < 0xf0 ) {
    return remains_80e0[((event-0x80)>>4)&0x0f];
  } else if ( event < 0xf7 ) {
    return remains_f0f6[event-0xf0];
  } else {
    return 1;
  }
}
