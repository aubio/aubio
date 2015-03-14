#! /usr/bin/env python
# encoding: utf-8
# WARNING! Do not edit! http://waf.googlecode.com/git/docs/wafbook/single.html#_obtaining_the_waf_file

import os,sys
from waflib.Tools import ccroot,ar,gxx
from waflib.Configure import conf
@conf
def find_clangxx(conf):
	cxx=conf.find_program('clang++',var='CXX')
	conf.get_cc_version(cxx,clang=True)
	conf.env.CXX_NAME='clang'
def configure(conf):
	conf.find_clangxx()
	conf.find_ar()
	conf.gxx_common_flags()
	conf.gxx_modifier_platform()
	conf.cxx_load_tools()
	conf.cxx_add_flags()
	conf.link_add_flags()
