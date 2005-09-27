/* 
 *
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

/* this file originally taken from FluidSynth - A Software Synthesizer
 * Copyright (C) 2003  Peter Hanappe and others.
 *
 *  \note fixed some track names causing segfault
 */

#include "aubio_priv.h"
#include "midi.h"
#include "midi_event.h"
#include "midi_track.h"
#include "midi_player.h"
#include "midi_file.h"


/** aubio_midi_file */
struct  _aubio_midi_file_t{
  //aubio_file_t fp;
  FILE *fp;
  int running_status;
  int c;
  int type;
  int ntracks;
  int uses_smpte;
  unsigned int smpte_fps;
  unsigned int smpte_res;
  unsigned int division;       /* If uses_SMPTE == 0 then division is 
				  ticks per beat (quarter-note) */
  double tempo;                /* Beats per second (SI rules =) */
  int tracklen;
  int trackpos;
  int eot;
  int varlen;
};

/***************************************************************
 *
 *                      MIDIFILE
 */

/** new_aubio_midi_file */
aubio_midi_file_t * new_aubio_midi_file(char* filename)
{
  aubio_midi_file_t* mf;

  mf = AUBIO_NEW(aubio_midi_file_t);
  if (mf == NULL) {
    AUBIO_ERR( "Out of memory");
    return NULL;
  }
  AUBIO_MEMSET(mf, 0, sizeof(aubio_midi_file_t));

  mf->c = -1;
  mf->running_status = -1;
  mf->fp = AUBIO_FOPEN(filename, "rb");

  if (mf->fp == NULL) {
    AUBIO_ERR( "Couldn't open the MIDI file !\n");
    AUBIO_FREE(mf);
    return NULL;    
  }

  if (aubio_midi_file_read_mthd(mf) != AUBIO_OK) {
    AUBIO_FREE(mf);
    return NULL;
  }
  return mf;
}

/** del_aubio_midi_file */
void del_aubio_midi_file(aubio_midi_file_t* mf)
{
  if (mf == NULL) {
    return;
  }
  if (mf->fp != NULL) {
    AUBIO_FCLOSE(mf->fp);
  }
  AUBIO_FREE(mf);
  return;
}

/** aubio_midi_file_getc */
int aubio_midi_file_getc(aubio_midi_file_t* mf)
{
  unsigned char c;
  int n;
  if (mf->c >= 0) {
    c = mf->c;
    mf->c = -1;
  } else {
    n = AUBIO_FREAD(&c, 1, 1, mf->fp);
    mf->trackpos++;
  }
  return (int) c;
}

/** aubio_midi_file_push */
int aubio_midi_file_push(aubio_midi_file_t* mf, int c)
{
  mf->c = c;
  return AUBIO_OK;
}

/** aubio_midi_file_read */
int aubio_midi_file_read(aubio_midi_file_t* mf, void* buf, int len)
{
  int num = AUBIO_FREAD(buf, 1, len, mf->fp);
  mf->trackpos += num;
#if DEBUG
  if (num != len) {
    AUBIO_DBG( "Coulnd't read the requested number of bytes");
  }
#endif
  return (num != len)? AUBIO_FAIL : AUBIO_OK;
}

/** aubio_midi_file_skip */
int aubio_midi_file_skip(aubio_midi_file_t* mf, int skip)
{
  int err = AUBIO_FSEEK(mf->fp, skip, SEEK_CUR);
  if (err) {
    AUBIO_ERR( "FAIL to seek position in file");
    return AUBIO_FAIL;    
  }
  return AUBIO_OK;
}

/** aubio_midi_file_read_mthd */
int aubio_midi_file_read_mthd(aubio_midi_file_t* mf)
{
  signed char mthd[15];
  if (aubio_midi_file_read(mf, mthd, 14) != AUBIO_OK) {
    return AUBIO_FAIL;
  }
  if ((AUBIO_STRNCMP((const char*)mthd, "MThd", 4) != 0) || (mthd[7] != 6) || (mthd[9] > 2)) {
    AUBIO_ERR( "Doesn't look like a MIDI file: invalid MThd header");
    return AUBIO_FAIL;
  }
  mf->type = mthd[9];
  mf->ntracks = (unsigned) mthd[11];
  mf->ntracks += (unsigned int) (mthd[10]) << 16;
  /** \bug: smpte timing not yet implemented */ 
  if(!mthd[12]){
  /*if((mthd[12]) < 0){*/
    mf->uses_smpte = 1;
    mf->smpte_fps = -mthd[12];
    mf->smpte_res = (unsigned) mthd[13];
    AUBIO_ERR( "File uses SMPTE timing -- Not implemented yet");
    return AUBIO_FAIL;
  } else {
    mf->uses_smpte = 0;
    mf->division = (mthd[12] << 8) | (mthd[13] & 0xff);
  }
  return AUBIO_OK;
}

/** aubio_midi_file_load_tracks */
int aubio_midi_file_load_tracks(aubio_midi_file_t* mf, aubio_midi_player_t* player)
{
  int i;
  for (i = 0; i < mf->ntracks; i++) {
    if (aubio_midi_file_read_track(mf, player, i) != AUBIO_OK) {
      return AUBIO_FAIL;
    }
  }
  return AUBIO_OK;
}

/** aubio_midi_file_read_tracklen */
int aubio_midi_file_read_tracklen(aubio_midi_file_t* mf)
{
  unsigned char length[5];
  if (aubio_midi_file_read(mf, length, 4) != AUBIO_OK) {
    return AUBIO_FAIL;
  }
  mf->tracklen = aubio_getlength(length);
  mf->trackpos = 0;
  mf->eot = 0;
  return AUBIO_OK;
}

/** aubio_midi_file_eot */
int aubio_midi_file_eot(aubio_midi_file_t* mf)
{
#if DEBUG
  if (mf->trackpos > mf->tracklen) {
    printf("track overrun: %d > %d\n", mf->trackpos, mf->tracklen);
  }
#endif
  return mf->eot || (mf->trackpos >= mf->tracklen);
}

/** aubio_midi_file_read_track */
int aubio_midi_file_read_track(aubio_midi_file_t* mf, aubio_midi_player_t* player, int num)
{
  aubio_track_t* track;
  unsigned char id[5], length[5];
  int found_track = 0;
  int skip;

  AUBIO_DBG("Loading track %d\n",num);
  if (aubio_midi_file_read(mf, id, 4) != AUBIO_OK) {
    AUBIO_DBG("Failed loading track %d\n",num);
    return AUBIO_FAIL;
  }
  
  id[4]='\0';

  while (!found_track){

    if (aubio_isasciistring((char*) id) == 0) {
      AUBIO_ERR( "An non-ascii track header found, currupt file");
      return AUBIO_FAIL;

    } else if (strcmp((char*) id, "MTrk") == 0) {

      found_track = 1;

      if (aubio_midi_file_read_tracklen(mf) != AUBIO_OK) {
        return AUBIO_FAIL;
      }

      track = new_aubio_track(num);
      if (track == NULL) {
        AUBIO_ERR( "Out of memory");
        return AUBIO_FAIL;
      }

      while (!aubio_midi_file_eot(mf)) {
        if (aubio_midi_file_read_event(mf, track) != AUBIO_OK) {
          return AUBIO_FAIL;	  
        }
      }

      aubio_midi_player_add_track(player, track);
    } else {
      found_track = 0;
      if (aubio_midi_file_read(mf, length, 4) != AUBIO_OK) {
        return AUBIO_FAIL;
      }
      skip = aubio_getlength(length);
      /* fseek(mf->fp, skip, SEEK_CUR); */
      if (aubio_midi_file_skip(mf, skip) != AUBIO_OK) {
        return AUBIO_FAIL;
      }
    }
  }

  if (feof(mf->fp)) {
    AUBIO_ERR( "Unexpected end of file");
    return AUBIO_FAIL;
  }
  AUBIO_DBG("Loaded track %d\n",num);
  return AUBIO_OK;
}

/** aubio_midi_file_read_varlen */
int aubio_midi_file_read_varlen(aubio_midi_file_t* mf)
{ 
  int i;
  int c;
  mf->varlen = 0;
  for (i = 0;;i++) {
    if (i == 4) {
      AUBIO_ERR( "Invalid variable length number");
      return AUBIO_FAIL;
    }
    c = aubio_midi_file_getc(mf);
    if (c < 0) {
      AUBIO_ERR( "Unexpected end of file");
      return AUBIO_FAIL;
    }
    if (c & 0x80){
      mf->varlen |= (int) (c & 0x7F);
      mf->varlen <<= 7;
    } else {
      mf->varlen += c;
      break;
    }    
  }
  return AUBIO_OK;
}

/** aubio_midi_file_read_event */
int aubio_midi_file_read_event(aubio_midi_file_t* mf, aubio_track_t* track)
{
  int dtime;
  int status;
  int type;
  int tempo;
  unsigned char* metadata = NULL;
  unsigned char* dyn_buf = NULL;
  unsigned char static_buf[256];
  int nominator, denominator, clocks, notes, sf, mi;
  aubio_midi_event_t* evt;
  int channel = 0;
  int param1 = 0;
  int param2 = 0;


  /* read the delta-time of the event */
  if (aubio_midi_file_read_varlen(mf) != AUBIO_OK) {
    return AUBIO_FAIL;
  }
  dtime = mf->varlen;

  /* read the status byte */
  status = aubio_midi_file_getc(mf);
  if (status < 0) {
    AUBIO_ERR( "Unexpected end of file");
    return AUBIO_FAIL;
  }

  /* not a valid status byte: use the running status instead */
  if ((status & 0x80) == 0) {
    if ((mf->running_status & 0x80) == 0) {
      AUBIO_ERR( "Undefined status and invalid running status");
      return AUBIO_FAIL;
    }
    aubio_midi_file_push(mf, status);
    status = mf->running_status;
  } 

  /* check what message we have */
  if (status & 0x80) {
    mf->running_status = status;

    if ((status == MIDI_SYSEX) || (status == MIDI_EOX)) {     /* system exclusif */
      /** \bug Sysex messages are not handled yet */
      /* read the length of the message */
      if (aubio_midi_file_read_varlen(mf) != AUBIO_OK) {
        return AUBIO_FAIL;
      }

      if (mf->varlen < 255) {
        metadata = &static_buf[0];
      } else {
        AUBIO_DBG( "%s: %d: alloc metadata, len = %d", __FILE__, __LINE__, mf->varlen);
        dyn_buf = AUBIO_MALLOC(mf->varlen + 1);
        if (dyn_buf == NULL) {
          //AUBIO_LOG(AUBIO_PANIC, "Out of memory");
          AUBIO_ERR("Out of memory");
          return AUBIO_FAIL;
        }
        metadata = dyn_buf;
      }

      /* read the data of the message */
      if (mf->varlen) {

        if (aubio_midi_file_read(mf, metadata, mf->varlen) != AUBIO_OK) {
          if (dyn_buf) {
            AUBIO_FREE(dyn_buf);
          }
          return AUBIO_FAIL;
        }

        if (dyn_buf) {
          AUBIO_DBG( "%s: %d: free metadata", __FILE__, __LINE__);
          AUBIO_FREE(dyn_buf);
        }
      }

      return AUBIO_OK;

    } else if (status == MIDI_META_EVENT) {             /* meta events */

      int result = AUBIO_OK;

      /* get the type of the meta message */
      type = aubio_midi_file_getc(mf);
      if (type < 0) {
        AUBIO_ERR( "Unexpected end of file");
        return AUBIO_FAIL;
      }

      /* get the length of the data part */
      if (aubio_midi_file_read_varlen(mf) != AUBIO_OK) {
        return AUBIO_FAIL;
      }

      if (mf->varlen) {

        if (mf->varlen < 255) {
          metadata = &static_buf[0];
        } else {
          AUBIO_DBG( "%s: %d: alloc metadata, len = %d", __FILE__, __LINE__, mf->varlen);
          dyn_buf = AUBIO_MALLOC(mf->varlen + 1);
          if (dyn_buf == NULL) {
            AUBIO_ERR("Out of memory");
            return AUBIO_FAIL;
          }
          metadata = dyn_buf;
        }

        /* read the data */
        if (aubio_midi_file_read(mf, metadata, mf->varlen) != AUBIO_OK) {
          if (dyn_buf) {
            AUBIO_FREE(dyn_buf);
          }
          return AUBIO_FAIL;
        }
      }

      /* handle meta data */
      switch (type) {

        case MIDI_COPYRIGHT:
          metadata[mf->varlen] = 0;
          break;

        case MIDI_TRACK_NAME:
          if (metadata != NULL) /* avoids crashes on empty tracks */
            metadata[mf->varlen] = 0;
          aubio_track_set_name(track, (char*) metadata);
          break;

        case MIDI_INST_NAME:
          metadata[mf->varlen] = 0;
          break;

        case MIDI_LYRIC:
          break;

        case MIDI_MARKER:
          break;

        case MIDI_CUE_POINT:
          break; /* don't care much for text events */

        case MIDI_EOT:
          if (mf->varlen != 0) {
            AUBIO_ERR("Invalid length for EndOfTrack event");
            result = AUBIO_FAIL;
            break;
          }
          mf->eot = 1;
          break; 

        case MIDI_SET_TEMPO:
          if (mf->varlen != 3) {
            AUBIO_ERR("Invalid length for SetTempo meta event");
            result = AUBIO_FAIL;
            break;
          }
          tempo = (metadata[0] << 16) + (metadata[1] << 8) + metadata[2];
          evt = new_aubio_midi_event();
          if (evt == NULL) {
            AUBIO_ERR( "Out of memory");
            result = AUBIO_FAIL;
            break;
          }
          evt->dtime = dtime;
          evt->type = MIDI_SET_TEMPO;
          evt->channel = 0;
          evt->param1 = tempo;
          evt->param2 = 0;
          aubio_track_add_event(track, evt);
          break; 

        case MIDI_SMPTE_OFFSET:
          if (mf->varlen != 5) {
            AUBIO_ERR("Invalid length for SMPTE Offset meta event");
            result = AUBIO_FAIL;
            break;
          }
          break; /* we don't use smtp */	

        case MIDI_TIME_SIGNATURE:
          if (mf->varlen != 4) {
            AUBIO_ERR("Invalid length for TimeSignature meta event");
            result = AUBIO_FAIL;
            break;
          }
          nominator = metadata[0];
          denominator = pow(2.0, (double) metadata[1]);
          clocks = metadata[2];
          notes = metadata[3];

          AUBIO_DBG("signature=%d/%d, metronome=%d, 32nd-notes=%d\n", 
              nominator, denominator, clocks, notes);

          break;

        case MIDI_KEY_SIGNATURE:
          if (mf->varlen != 2) {
            AUBIO_ERR( "Invalid length for KeySignature meta event");
            result = AUBIO_FAIL;
            break;
          }
          sf = metadata[0];
          mi = metadata[1];
          break;

        case MIDI_SEQUENCER_EVENT:
          AUBIO_DBG("Sequencer event ignored\n");
          break;

        default:
          break;
      }

      if (dyn_buf) {
        AUBIO_DBG( "%s: %d: free metadata", __FILE__, __LINE__);
        AUBIO_FREE(dyn_buf);
      }

      return result;

    } else {                /* channel messages */

      type = status & 0xf0;
      channel = status & 0x0f;

      /* all channel message have at least 1 byte of associated data */
      if ((param1 = aubio_midi_file_getc(mf)) < 0) {
        AUBIO_ERR( "Unexpected end of file");
        return AUBIO_FAIL;
      }

      switch (type) {

        case NOTE_ON:
          if ((param2 = aubio_midi_file_getc(mf)) < 0) {
            AUBIO_ERR( "Unexpected end of file");
            return AUBIO_FAIL;
          }
          break;

        case NOTE_OFF:	
          if ((param2 = aubio_midi_file_getc(mf)) < 0) {
            AUBIO_ERR( "Unexpected end of file");
            return AUBIO_FAIL;
          }
          break;

        case KEY_PRESSURE:
          if ((param2 = aubio_midi_file_getc(mf)) < 0) {
            AUBIO_ERR( "Unexpected end of file");
            return AUBIO_FAIL;
          }
          break;

        case CONTROL_CHANGE:
          if ((param2 = aubio_midi_file_getc(mf)) < 0) {
            AUBIO_ERR( "Unexpected end of file");
            return AUBIO_FAIL;
          }
          break;

        case PROGRAM_CHANGE:
          break;

        case CHANNEL_PRESSURE:
          break;

        case PITCH_BEND:
          if ((param2 = aubio_midi_file_getc(mf)) < 0) {
            AUBIO_ERR( "Unexpected end of file");
            return AUBIO_FAIL;
          }

          param1 = ((param2 & 0x7f) << 7) | (param1 & 0x7f);
          param2 = 0;
          break;

        default:
          /* Can't possibly happen !? */
          AUBIO_ERR( "Unrecognized MIDI event");
          return AUBIO_FAIL;      
      }
      evt = new_aubio_midi_event();
      if (evt == NULL) {
        AUBIO_ERR( "Out of memory");
        return AUBIO_FAIL;
      }
      evt->dtime = dtime;
      evt->type = type;
      evt->channel = channel;
      evt->param1 = param1;
      evt->param2 = param2;
      aubio_track_add_event(track, evt);
    }
  }
  return AUBIO_OK;
}

/** aubio_midi_file_get_division */
int aubio_midi_file_get_division(aubio_midi_file_t* midifile)
{
  return midifile->division;
}


/** aubio_isasciistring */
int aubio_isasciistring(char* s)
{
  int i;
  int len = (int) AUBIO_STRLEN(s);
  for (i = 0; i < len; i++) {
    if (!aubio_isascii(s[i])) {
      return 0;
    }
  }
  return 1;
}

/** aubio_getlength */
long aubio_getlength(unsigned char *s)
{
  long i = 0;
  i = s[3] | (s[2]<<8) | (s[1]<<16) | (s[0]<<24);
  return i;
}

