 #
 # Copyright 2004 Apache Software Foundation 
 # 
 # Licensed under the Apache License, Version 2.0 (the "License"); you
 # may not use this file except in compliance with the License.  You
 # may obtain a copy of the License at
 #
 #      http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 # implied.  See the License for the specific language governing
 # permissions and limitations under the License.
 #
 # Originally developed by Gregory Trubetskoy.
 #
 # $Id: publisher.py,v 1.36 2004/02/16 19:47:27 grisha Exp $

"""
  This handler is conceputally similar to Zope's ZPublisher, except
  that it:

  1. Is written specifically for mod_python and is therefore much faster
  2. Does not require objects to have a documentation string
  3. Passes all arguments as simply string
  4. Does not try to match Python errors to HTTP errors
  5. Does not give special meaning to '.' and '..'.

  This is a modified version of mod_python.publisher.handler Only the first
  directory argument is matched, the rest is left for path_info. A default
  one must be provided.

"""

from mod_python import apache
from mod_python import util
from mod_python.publisher import resolve_object,process_auth,imp_suffixes

import sys
import os
import re

from types import *

def configure_handler(req,default):

    req.allow_methods(["GET", "POST"])
    if req.method not in ["GET", "POST"]:
        raise apache.SERVER_RETURN, apache.HTTP_METHOD_NOT_ALLOWED

    func_path = ""
    if req.path_info:
        func_path = req.path_info[1:] # skip first /
        #func_path = func_path.replace("/", ".")
        #if func_path[-1:] == ".":
        #    func_path = func_path[:-1] 
        # changed: only keep the first directory
        func_path = re.sub('/.*','',func_path)

    # default to 'index' if no path_info was given
    if not func_path:
        func_path = "index"

    # if any part of the path begins with "_", abort
    if func_path[0] == '_' or func_path.count("._"):
        raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND

    ## import the script
    path, module_name =  os.path.split(req.filename)
    if not module_name:
        module_name = "index"

    # get rid of the suffix
    #   explanation: Suffixes that will get stripped off
    #   are those that were specified as an argument to the
    #   AddHandler directive. Everything else will be considered
    #   a package.module rather than module.suffix
    exts = req.get_addhandler_exts()
    if not exts:
        # this is SetHandler, make an exception for Python suffixes
        exts = imp_suffixes
    if req.extension:  # this exists if we're running in a | .ext handler
        exts += req.extension[1:] 
    if exts:
        suffixes = exts.strip().split()
        exp = "\\." + "$|\\.".join(suffixes)
        suff_matcher = re.compile(exp) # python caches these, so its fast
        module_name = suff_matcher.sub("", module_name)

    # import module (or reload if needed)
    # the [path] argument tells import_module not to allow modules whose
    # full path is not in [path] or below.
    config = req.get_config()
    autoreload=int(config.get("PythonAutoReload", 1))
    log=int(config.get("PythonDebug", 0))
    try:
        module = apache.import_module(module_name,
                                      autoreload=autoreload,
                                      log=log,
                                      path=[path])
    except ImportError:
        et, ev, etb = sys.exc_info()
        # try again, using default module, perhaps this is a
        # /directory/function (as opposed to /directory/module/function)
        func_path = module_name
        module_name = "index"
        try:
            module = apache.import_module(module_name,
                                          autoreload=autoreload,
                                          log=log,
                                          path=[path])
        except ImportError:
            # raise the original exception
            raise et, ev, etb
        
    # does it have an __auth__?
    realm, user, passwd = process_auth(req, module)

    # resolve the object ('traverse')
    try:
        object = resolve_object(req, module, func_path, realm, user, passwd)
    except AttributeError:
        # changed, return the default path instead
        #raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND
        object = default
    # not callable, a class or an unbound method
    if (not callable(object) or 
        type(object) is ClassType or
        (hasattr(object, 'im_self') and not object.im_self)):

        result = str(object)
        
    else:
        # callable, (but not a class or unbound method)
        
        # process input, if any
        req.form = util.FieldStorage(req, keep_blank_values=1)
        
        result = util.apply_fs_data(object, req.form, req=req)

    if result or req.bytes_sent > 0 or req.next:
        
        if result is None:
            result = ""
        else:
            result = str(result)

        # unless content_type was manually set, we will attempt
        # to guess it
        if not req._content_type_set:
            # make an attempt to guess content-type
            if result[:100].strip()[:6].lower() == '<html>' \
               or result.find('</') > 0:
                req.content_type = 'text/html'
            else:
                req.content_type = 'text/plain'

        if req.method != "HEAD":
            req.write(result)
        else:
            req.write("")
        return apache.OK
    else:
        req.log_error("mod_python.publisher: %s returned nothing." % `object`)
        return apache.HTTP_INTERNAL_SERVER_ERROR

