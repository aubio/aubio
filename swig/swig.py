#! /usr/bin/env python
# encoding: UTF-8
# Petar Forai
# Thomas Nagy 2008

import re
import Task, Utils, Logs
from TaskGen import extension
from Configure import conf
import preproc

SWIG_EXTS = ['.swig', '.i']

swig_str = '${SWIG} ${SWIGFLAGS} ${SRC}'
cls = Task.simple_task_type('swig', swig_str, color='BLUE', before='cc cxx')

re_module = re.compile('%module(?:\s*\(.*\))?\s+(.+)', re.M)

re_1 = re.compile(r'^%module.*?\s+([\w]+)\s*?$', re.M)
re_2 = re.compile('%include "(.*)"', re.M)
re_3 = re.compile('#include "(.*)"', re.M)

def scan(self):
	"scan for swig dependencies, climb the .i files"
	env = self.env

	lst_src = []

	seen = []
	to_see = [self.inputs[0]]

	while to_see:
		node = to_see.pop(0)
		if node.id in seen:
			continue
		seen.append(node.id)
		lst_src.append(node)

		# read the file
		code = node.read(env)
		code = preproc.re_nl.sub('', code)
		code = preproc.re_cpp.sub(preproc.repl, code)

		# find .i files and project headers
		names = re_2.findall(code) + re_3.findall(code)
		for n in names:
			for d in self.generator.swig_dir_nodes + [node.parent]:
				u = d.find_resource(n)
				if u:
					to_see.append(u)
					break
			else:
				Logs.warn('could not find %r' % n)

	# list of nodes this one depends on, and module name if present
	if Logs.verbose:
		Logs.debug('deps: deps for %s: %s' % (str(self), str(lst_src)))
	return (lst_src, [])
cls.scan = scan

# provide additional language processing
swig_langs = {}
def swig(fun):
	swig_langs[fun.__name__.replace('swig_', '')] = fun

@swig
def swig_python(tsk):
	tsk.set_outputs(tsk.inputs[0].parent.find_or_declare(tsk.module + '.py'))

@swig
def swig_ocaml(tsk):
	tsk.set_outputs(tsk.inputs[0].parent.find_or_declare(tsk.module + '.ml'))
	tsk.set_outputs(tsk.inputs[0].parent.find_or_declare(tsk.module + '.mli'))

def add_swig_paths(self):
	if getattr(self, 'add_swig_paths_done', None):
		return
	self.add_swig_paths_done = True

	self.swig_dir_nodes = []
	for x in self.to_list(self.includes):
		node = self.path.find_dir(x)
		if not node:
			Logs.warn('could not find the include %r' % x)
			continue
		self.swig_dir_nodes.append(node)

	# add the top-level, it is likely to be added
	self.swig_dir_nodes.append(self.bld.srcnode)
	for x in self.swig_dir_nodes:
		self.env.append_unique('SWIGFLAGS', '-I%s' % x.abspath(self.env)) # build dir
		self.env.append_unique('SWIGFLAGS', '-I%s' % x.abspath()) # source dir

@extension(SWIG_EXTS)
def i_file(self, node):
	flags = self.to_list(getattr(self, 'swig_flags', []))

	ext = '.swigwrap_%d.c' % self.idx
	if '-c++' in flags:
		ext += 'xx'

	# the user might specify the module directly
	module = getattr(self, 'swig_module', None)
	if not module:
		# else, open the files and search
		txt = node.read(self.env)
		m = re_module.search(txt)
		if not m:
			raise "for now we are expecting a module name in the main swig file"
		module = m.group(1)
	out_node = node.parent.find_or_declare(module + ext)

	# the task instance
	tsk = self.create_task('swig')
	tsk.set_inputs(node)
	tsk.set_outputs(out_node)
	tsk.module = module
	tsk.env['SWIGFLAGS'] = flags

	if not '-outdir' in flags:
		flags.append('-outdir')
		flags.append(node.parent.abspath(self.env))

	if not '-o' in flags:
		flags.append('-o')
		flags.append(out_node.abspath(self.env))

	# add the language-specific output files as nodes
	# call funs in the dict swig_langs
	for x in flags:
		# obtain the language
		x = x[1:]
		try:
			fun = swig_langs[x]
		except KeyError:
			pass
		else:
			fun(tsk)

	self.allnodes.append(out_node)

	add_swig_paths(self)

@conf
def check_swig_version(conf, minver=None):
	"""Check for a minimum swig version  like conf.check_swig_version('1.3.28')
	or conf.check_swig_version((1,3,28)) """
	reg_swig = re.compile(r'SWIG Version\s(.*)', re.M)

	swig_out = Utils.cmd_output('%s -version' % conf.env['SWIG'])

	swigver = [int(s) for s in reg_swig.findall(swig_out)[0].split('.')]
	if isinstance(minver, basestring):
		minver = [int(s) for s in minver.split(".")]
	if isinstance(minver, tuple):
		minver = [int(s) for s in minver]
	result = (minver is None) or (minver[:3] <= swigver[:3])
	swigver_full = '.'.join(map(str, swigver))
	if result:
		conf.env['SWIG_VERSION'] = swigver_full
	minver_str = '.'.join(map(str, minver))
	if minver is None:
		conf.check_message_custom('swig version', '', swigver_full)
	else:
		conf.check_message('swig version', '>= %s' % (minver_str,), result, option=swigver_full)
	return result

def detect(conf):
	swig = conf.find_program('swig', var='SWIG', mandatory=True)

