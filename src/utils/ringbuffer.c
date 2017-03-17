

#include "aubio_priv.h"
#include "fvec.h"
#include "utils/ringbuffer.h"
#include <assert.h>

struct _aubio_ringbuffer_t
{
  uint_t maxwrite;
  uint_t maxrequest;
  fvec_t *buffer;
  uint_t write_pos;
  uint_t read_pos;
  sint_t available;
};

aubio_ringbuffer_t * new_aubio_ringbuffer(uint_t maxwrite, uint_t maxrequest)
{
  aubio_ringbuffer_t *p = AUBIO_NEW (aubio_ringbuffer_t);
  p->maxwrite = maxwrite;
  p->maxrequest = maxrequest;
  //p->buffer = new_fvec(MAX(maxwrite, maxrequest));
  p->buffer = new_fvec(maxwrite + maxrequest);
  p->write_pos = p->read_pos = p->available = 0;
  return p;
}

uint_t aubio_ringbuffer_push(aubio_ringbuffer_t *o, fvec_t *input, uint_t write) {
  if ((sint_t)o->buffer->length < o->available + (sint_t)write) {
    AUBIO_ERR("ringbuffer: push: writing %d but capacity is %d and %d are available\n",
        write, o->buffer->length, o->available);
    return AUBIO_FAIL;
  }
  if (write > o->maxwrite) {
    AUBIO_ERR("ringbuffer: push: trying to write %d but maxwrite is %d\n",
        write, o->maxwrite);
    return AUBIO_FAIL;
  } else
  if (write > input->length) {
    AUBIO_ERR("ringbuffer: push: trying to write %d, but input vector is %d long\n",
        write, input->length);
    return AUBIO_FAIL;
  } else
  if (o->write_pos + write <= o->buffer->length) {
    // write everything at once
    fvec_t tmp, tmpin;
    tmp.data = o->buffer->data + o->write_pos; tmp.length = write;
    tmpin.data = input->data; tmpin.length = write;
    //assert(tmpin.length == tmp.length);
    fvec_copy(&tmpin, &tmp);

    //AUBIO_WRN("ringbuffer: push1: changing write_pos from %d\n", o->write_pos);
    o->write_pos = (o->write_pos % o->buffer->length);
    o->write_pos += write;
    o->available += write;
    //AUBIO_WRN("ringbuffer: push1: changed write_pos to %d\n", o->write_pos);

    //AUBIO_WRN("ringbuffer: push1: wrote %d, %d available\n", write,
    //    o->available);

    return AUBIO_OK;
  } else {
    // write in two folds
    uint_t remaining = o->buffer->length - o->write_pos;
    fvec_t tmp, tmpin;
    // write end
    if (remaining) {
      tmp.data = o->buffer->data + o->write_pos;
      tmp.length = remaining;
      tmpin.data = input->data;
      tmpin.length = remaining;
      //assert(tmpin.length == tmp.length);
      fvec_copy(&tmpin, &tmp);
    }
    // write start
    tmp.data = o->buffer->data;
    tmp.length = write - remaining;
    tmpin.data = input->data + remaining;
    tmpin.length = write - remaining;
    //assert(tmpin.length == tmp.length);
    fvec_copy(&tmpin, &tmp);
    //AUBIO_WRN("ringbuffer: push2: changing write_pos from %d\n", o->write_pos);
    o->write_pos += write;
    o->write_pos = (o->write_pos % o->buffer->length);
    o->available += write;
    //AUBIO_WRN("ringbuffer: push2: changed write_pos to %d\n", o->write_pos);

    //AUBIO_WRN("ringbuffer: push2: wrote %d, %d available\n", write,
    //    o->available);

    return AUBIO_OK;
  }
}

uint_t aubio_ringbuffer_pull(aubio_ringbuffer_t *o, fvec_t *output, uint_t request) {
  if (o->available < (sint_t)request) {
    AUBIO_ERR("ringbuffer: pull: requested %d but %d available\n",
        request, o->available);
    return AUBIO_FAIL;
  }
  if (request > o->maxrequest) {
    AUBIO_ERR("ringbuffer: pull: trying to request %d but maxrequest is %d\n",
        request, o->maxrequest);
    return AUBIO_FAIL;
  } else
  if (request > output->length) {
    AUBIO_ERR("ringbuffer: pull: trying to request %d, but output vector is %d long\n",
        request, output->length);
    return AUBIO_FAIL;
  } else
  if (o->read_pos + request <= o->buffer->length) {
    // read everything at once
    fvec_t tmp, tmpout;
    tmp.data = o->buffer->data + o->read_pos; tmp.length = request;
    tmpout.data = output->data; tmpout.length = request;
    //assert(tmpout.length == tmp.length);
    fvec_copy(&tmp, &tmpout);
    //AUBIO_WRN("ringbuffer: pull1: changing read_pos from %d\n", o->read_pos);
    o->read_pos += request;
    o->read_pos %= o->buffer->length;
    o->available -= request;
    //AUBIO_WRN("ringbuffer: pull1: read %d, %d available\n", request,
    //    o->available);
    //AUBIO_WRN("ringbuffer: pull1: changed read_pos to %d\n", o->read_pos);
    return AUBIO_OK;
  } else {
    // read in two folds
    uint_t remaining = o->buffer->length - o->read_pos;
    fvec_t tmp, tmpout;
    tmp.data = o->buffer->data + o->read_pos;
    tmp.length = remaining;
    tmpout.data = output->data;
    tmpout.length = remaining;
    //assert(tmpout.length == tmp.length);
    fvec_copy(&tmpout, &tmp);
    // write start
    tmp.data = o->buffer->data;
    tmp.length = request - remaining;
    tmpout.data = output->data + remaining;
    tmpout.length = request - remaining;
    //assert(tmpout.length == tmp.length);
    fvec_copy(&tmp, &tmpout);
    //AUBIO_WRN("ringbuffer: pull2: changing read_pos from %d\n", o->read_pos);
    o->read_pos += request;
    o->read_pos %= o->buffer->length;
    o->available -= request;
    //AUBIO_WRN("ringbuffer: pull2: changed read_pos to %d\n", o->read_pos);
    //AUBIO_WRN("ringbuffer: pull2: read %d, %d available\n", request,
    //    o->available);
    return AUBIO_OK;
  }
}

uint_t aubio_ringbuffer_reset(aubio_ringbuffer_t *o) {
  o->read_pos = o->write_pos = 0;
  o->available = 0;
  return AUBIO_OK;
}

sint_t aubio_ringbuffer_get_available(aubio_ringbuffer_t *o) {
  //AUBIO_WRN("ringbuffer: got %d available (%d ... %d)\n",
  //    o->available, o->read_pos, o->write_pos);
  return o->available;
}

void del_aubio_ringbuffer(aubio_ringbuffer_t *o) {
  del_fvec(o->buffer);
  AUBIO_FREE(o);
}
