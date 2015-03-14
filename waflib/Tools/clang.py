#! /usr/bin/env python
# encoding: utf-8
# WARNING! Do not edit! http://waf.googlecode.com/git/docs/wafbook/single.html#_obtaining_the_waf_file

import os,sys
from waflib.Tools import ccroot,ar,gcc
from waflib.Configure import conf
@conf
def find_clang(conf):
	cc=conf.find_program('clang',var='CC')
	conf.get_cc_version(cc,clang=True)
	conf.env.CC_NAME='clang'
def configure(conf):
	conf.find_clang()
	conf.find_ar()
	conf.gcc_common_flags()
	conf.gcc_modifier_platform()
	conf.cc_load_tools()
	conf.cc_add_flags()
	conf.link_add_flags()
