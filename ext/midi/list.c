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

/* Modified by the GLib Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

#include "aubio_priv.h"
#include "list.h"



aubio_list_t*
new_aubio_list(void)
{
  aubio_list_t* list;
  list = AUBIO_NEW(aubio_list_t);
  list->data = NULL;
  list->next = NULL;
  return list;
}

void
del_aubio_list(aubio_list_t *list)
{
  aubio_list_t *next;
  while (list) {
    next = list->next;
    AUBIO_FREE(list);
    list = next;
  }
}

void
del_aubio_list1(aubio_list_t *list)
{
  if (list) {
    AUBIO_FREE(list);
  }
}

aubio_list_t*
aubio_list_append(aubio_list_t *list, void*  data)
{
  aubio_list_t *new_list;
  aubio_list_t *last;

  new_list = new_aubio_list();
  new_list->data = data;

  if (list)
    {
      last = aubio_list_last(list);
      /* g_assert (last != NULL); */
      last->next = new_list;

      return list;
    }
  else
      return new_list;
}

aubio_list_t*
aubio_list_prepend(aubio_list_t *list, void* data)
{
  aubio_list_t *new_list;

  new_list = new_aubio_list();
  new_list->data = data;
  new_list->next = list;

  return new_list;
}

aubio_list_t*
aubio_list_nth(aubio_list_t *list, int n)
{
  while ((n-- > 0) && list) {
    list = list->next;
  }

  return list;
}

aubio_list_t*
aubio_list_remove(aubio_list_t *list, void* data)
{
  aubio_list_t *tmp;
  aubio_list_t *prev;

  prev = NULL;
  tmp = list;

  while (tmp) {
    if (tmp->data == data) {
      if (prev) {
	prev->next = tmp->next;
      }
      if (list == tmp) {
	list = list->next;
      }
      tmp->next = NULL;
      del_aubio_list(tmp);
      
      break;
    }
    
    prev = tmp;
    tmp = tmp->next;
  }

  return list;
}

aubio_list_t*
aubio_list_remove_link(aubio_list_t *list, aubio_list_t *link)
{
  aubio_list_t *tmp;
  aubio_list_t *prev;

  prev = NULL;
  tmp = list;

  while (tmp) {
    if (tmp == link) {
      if (prev) {
	prev->next = tmp->next;
      }
      if (list == tmp) {
	list = list->next;
      }
      tmp->next = NULL;
      break;
    }
    
    prev = tmp;
    tmp = tmp->next;
  }
  
  return list;
}

static aubio_list_t* 
aubio_list_sort_merge(aubio_list_t *l1, aubio_list_t *l2, aubio_compare_func_t compare_func)
{
  aubio_list_t list, *l;

  l = &list;

  while (l1 && l2) {
    if (compare_func(l1->data,l2->data) < 0) {
      l = l->next = l1;
      l1 = l1->next;
    } else {
      l = l->next = l2;
      l2 = l2->next;
    }
  }
  l->next= l1 ? l1 : l2;
  
  return list.next;
}

aubio_list_t* 
aubio_list_sort(aubio_list_t *list, aubio_compare_func_t compare_func)
{
  aubio_list_t *l1, *l2;

  if (!list) {
    return NULL;
  }
  if (!list->next) {
    return list;
  }

  l1 = list; 
  l2 = list->next;

  while ((l2 = l2->next) != NULL) {
    if ((l2 = l2->next) == NULL) 
      break;
    l1=l1->next;
  }
  l2 = l1->next; 
  l1->next = NULL;

  return aubio_list_sort_merge(aubio_list_sort(list, compare_func),
			      aubio_list_sort(l2, compare_func),
			      compare_func);
}


aubio_list_t*
aubio_list_last(aubio_list_t *list)
{
  if (list) {
    while (list->next)
      list = list->next;
  }

  return list;
}

int 
aubio_list_size(aubio_list_t *list)
{
  int n = 0;
  while (list) {
    n++;
    list = list->next;
  }
  return n;
}

aubio_list_t* aubio_list_insert_at(aubio_list_t *list, int n, void* data)
{
  aubio_list_t *new_list;
  aubio_list_t *cur;
  aubio_list_t *prev = NULL;

  new_list = new_aubio_list();
  new_list->data = data;

  cur = list;
  while ((n-- > 0) && cur) {
    prev = cur;
    cur = cur->next;
  }

  new_list->next = cur;

  if (prev) {
    prev->next = new_list;    
    return list;
  } else {
    return new_list;
  }
}
