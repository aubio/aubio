#!/usr/bin/python

doc = """
This script works with mod_python to browse a collection of annotated wav files
and results.

you will need to have at least the following packages need to be installed (the
name of the command line tool is precised in parenthesis):

libapache-mod-python (apache2 prefered)
sndfile-programs (sndfile-info)
vorbis-tools (oggenc)
python-gnuplot
python-numpy

Try the command line tools in aubio/python to test your installation.

NOTE: this script is probably horribly insecure.

example configuration for apache to put in your preferred virtual host.

<Directory /home/piem/public_html/aubioweb>
    # Minimal config
    AddHandler mod_python .py
    # Disable these in production
    PythonDebug On
    PythonAutoReload on
    # Default handler in url
    PythonHandler aubioweb
    ## Authentication stuff (optional)
    #PythonAuthenHandler aubioweb
    #AuthType Basic
    #AuthName "Restricted Area"
    #require valid-user
    # make default listing
    DirectoryIndex aubioweb/
</Directory>

"""

from aubio.web.html import *

def handler(req):
    from aubio.web.browser import *
    from mod_python import Session
    req.sess = Session.Session(req)
    req.sess['login']='new aubio user'
    req.sess.save()
    return configure_handler(req,index)

def index(req,threshold='0.3'):
    navigation(req)
    print_command(req,"sfinfo %%i")
    return footer(req)

def show_info(req,verbose=''):
    navigation(req)
    print_command(req,"sndfile-info %%i")
    return footer(req)

def feedback(req):
    navigation(req)
    req.write("""
    Please provide feedback below:
  <p>                           
  <form action="/~piem/aubioweb/email" method="POST">
      Name:    <input type="text" name="name"><br>
      Email:   <input type="text" name="email"><br>
      Comment: <textarea name="comment" rows=4 cols=20></textarea><br>
      <input type="submit">
  </form>
    """)

WEBMASTER='piem@calabaza'
SMTP_SERVER='localhost'

def email(req,name,email,comment):
    import smtplib
    # make sure the user provided all the parameters
    if not (name and email and comment):
        return "A required parameter is missing, \
               please go back and correct the error"
    # create the message text
    msg = """\
From: %s                                                                                                                                           
Subject: feedback
To: %s

I have the following comment:

%s

Thank You,

%s

""" % (email, WEBMASTER, comment, name)
    # send it out
    conn = smtplib.SMTP(SMTP_SERVER)
    try:
	conn.sendmail(email, [WEBMASTER], msg)
    except smtplib.SMTPSenderRefused:
	return """<html>please provide a valid email</html>"""
	
    conn.quit()
    # provide feedback to the user
    s = """\
<html>
Dear %s,<br>
Thank You for your kind comments, we
will get back to you shortly.
</html>""" % name
    return s
