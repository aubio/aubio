

#ifndef AUBIO_RINGBUFFER_H
#define AUBIO_RINGBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_ringbuffer_t aubio_ringbuffer_t;

aubio_ringbuffer_t * new_aubio_ringbuffer(uint_t maxwrite, uint_t maxrequest);

uint_t aubio_ringbuffer_push(aubio_ringbuffer_t *o, fvec_t *output, uint_t write);

uint_t aubio_ringbuffer_pull(aubio_ringbuffer_t *o, fvec_t *input, uint_t request);

sint_t aubio_ringbuffer_get_available(aubio_ringbuffer_t *o);

uint_t aubio_ringbuffer_reset(aubio_ringbuffer_t *o);

void del_aubio_ringbuffer(aubio_ringbuffer_t *o);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_RINGBUFFER_H */
