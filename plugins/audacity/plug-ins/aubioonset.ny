;nyquist plug-in
;version 1
;type analyze
;name "Onset Detection..."
;action "Generate onset label track using aubioonset..."
;info "Aubio onset detector:\n Generates a label track with markers at the beginning of audio events"
;control threshold "Detection threshold" real "[0.001-0.900...]" 0.1 0.001 1.0

; largest number of samples 
(setf largenumber 1000000000) 
; some temporary files
(setf infile "/tmp/test.wav")
(setf tmfile "/tmp/test.txt")
; our command lines
(setf aubiocmd (strcat "aubioonset -t " (ftoa threshold) " -i " infile " > " tmfile))
(setf deletcmd "rm -f /tmp/test.wav /tmp/test.txt")

; save current selection in /tmp
; bug: should check the sound is mono
(s-save s (snd-length s largenumber) "/tmp/test.wav")
; run aubio
(system aubiocmd)

; read the file and build the list of label in result
(let* (
	(fp (open "/tmp/test.txt"  :direction :input))
 	(result '())
	(n 1)
 	(c (read-line fp))
      )
 (read-line fp)
 (while (not (not c))
   (setq result 
	(append
	result
	(list (list (atof c) ""))
 	))
   (setf c (read-line fp))
   (setq n (+ n 1))
  )
(close fp)
(system deletcmd)
;uncomment to debug
;(print result) 
result)
