;nyquist plug-in
;version 1
;type analyze
;name "Onset Detection..."
;action "Generate onset label track using aubioonset..."
;info "Aubio onset detector:\n Generates a label track with markers at the beginning of audio events"
;control dmode "Detection Beats/Onsets" int "0=Beats 1=Onsets" 1 0 1
;control threshold "Detection threshold" real "[0.001-0.900...]" 0.3 0.001 1.0
;control omode "Mode" int "0=ComplexDomain 1=HFC 2=Specdiff 3=Phase 4=Energy" 0 0 3

;Create a function to make the sum the two channels if they are stereo
(defun mono-s (s-in) 
  (if (arrayp s-in) (snd-add (aref s-in 0) (aref s-in 1)) s-in)
)

; path to aubio commands
(cond 
  ((= dmode 0)(setf aubiocmd "/home/piem/aubio/aubio/examples/aubiotrack"))
  (t (setf aubiocmd "/home/piem/aubio/aubio/examples/aubioonset"))
)

(cond 
  ((= omode 0)(setf onsetmode "complexdomain"))
  ((= omode 1)(setf onsetmode "hfc"))
  ((= omode 2)(setf onsetmode "specdiff"))
  (t (setf onsetmode "dual"))
)

; largest number of samples 
(setf largenumber 1000000000) 

; some temporary files
;(setf infile (system "mktmp tmp-aubio-XXXXXX"))
;(setf tmfile (system "mktmp tmp-aubio-XXXXXX"))
(setf infile "/tmp/aubio-insecure.wav")
(setf tmfile "/tmp/aubio-insecure.txt")

; our command lines
(setf aubiocmd (strcat 
        aubiocmd
        " -O " onsetmode
        " -t " (ftoa threshold) 
        " -i " infile 
        " > "  tmfile))
(setf deletcmd (strcat "rm -f " infile " " tmfile))

; save current sound selection into mono infile 
(s-save (mono-s s) (snd-length (mono-s s) largenumber) infile)

; run aubio
(system aubiocmd)

; read the file and build the list of label in result
(let* (
        (fp (open tmfile :direction :input))
        (result '())
        (n 1)
        (c (read-line fp))
      )
 (read-line fp)
 
 ;(setf oldc c)
 (while (not (not c))
   (setq result 
        (append
        result
        ;(list (list (strcat oldc "	" c) ""))
        (list (list (atof c) ""))
        ))
   ;(setf oldc c)
   (setf c (read-line fp))
   (setq n (+ n 1))
  )
  (close fp)

  (system deletcmd)

  ;uncomment to debug
  ;(print result) 


  ;If no silence markers were found, return a message
  (if (null result)
   (setq result "No onsets or beats found, no passages marked")
  )
  
  result

)

