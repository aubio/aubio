/*
	 Copyright (C) 2003 Paul Brossier

	 This program is free software; you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation; either version 2 of the License, or
	 (at your option) any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program; if not, write to the Free Software
	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
	 
*/

#define debug(...) if (verbose) fprintf (stderr, __VA_ARGS__)
#define errmsg(...) fprintf (stderr, __VA_ARGS__)
#define outmsg(...) fprintf (stdout, __VA_ARGS__)

extern int verbose;
extern int usejack;
extern int usedoubled;
extern const char * output_filename;
extern const char * input_filename;
/* defined in utils.c */
void usage (FILE * stream, int exit_code);
int parse_args (int argc, char **argv);

