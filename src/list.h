/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * \file
 * list of objects
 *
 * implement common list structure and its various functions
 * 
 * adapted for audio by Paul Brossier
 * 
 * Modified by the GLib Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

#ifndef _AUBIO_LIST_H
#define _AUBIO_LIST_H

typedef struct _aubio_list_t aubio_list_t;

struct _aubio_list_t
{
  void* data;
  aubio_list_t *next;
};

typedef int (*aubio_compare_func_t)(void* a, void* b);

aubio_list_t* new_aubio_list(void);
void del_aubio_list(aubio_list_t *list);
void del_aubio_list1(aubio_list_t *list);
aubio_list_t* aubio_list_sort(aubio_list_t *list, aubio_compare_func_t compare_func);
aubio_list_t* aubio_list_append(aubio_list_t *list, void* data);
aubio_list_t* aubio_list_prepend(aubio_list_t *list, void* data);
aubio_list_t* aubio_list_remove(aubio_list_t *list, void* data);
aubio_list_t* aubio_list_remove_link(aubio_list_t *list, aubio_list_t *llink);
aubio_list_t* aubio_list_nth(aubio_list_t *list, int n);
aubio_list_t* aubio_list_last(aubio_list_t *list);
aubio_list_t* aubio_list_insert_at(aubio_list_t *list, int n, void* data);
int aubio_list_size(aubio_list_t *list);

#define aubio_list_next(slist)	((slist) ? (((aubio_list_t *)(slist))->next) : NULL)
#define aubio_list_get(slist)	((slist) ? ((slist)->data) : NULL)

#endif  /* _AUBIO_LIST_H */
