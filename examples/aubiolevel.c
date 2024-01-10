/*
  Copyright (C) 2022 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "utils.h"
#include "parse_args.h"

smpl_t level;

void process_block(fvec_t * ibuf, fvec_t * obuf)
{
  (void)(obuf);
  level = aubio_db_spl (ibuf);
}

void process_print (void)
{
  print_time(blocks * hop_size);
  outmsg(" %f\n", level);
}

int main(int argc, char **argv) {
  int ret = 0;

  buffer_size = 2048;

  examples_common_init(argc,argv);

  verbmsg ("using source: %s at %dHz\n", source_uri, samplerate);

  verbmsg ("buffer_size: %d, ", buffer_size);
  verbmsg ("hop_size: %d, ", hop_size);

  examples_common_process(process_block, process_print);

  examples_common_del();
  return ret;
}
