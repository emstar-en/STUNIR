(in-package :stunir.generated)

(defun slurp-file (path)
  (with-open-file (in path :direction :input :external-format :utf-8)
    (let ((out (make-string-output-stream)))
      (loop for line = (read-line in nil nil) while line do
            (write-string line out)
            (write-char #\Newline out))
      (get-output-stream-string out))))
(defun main ()
  (write-string (slurp-file "payload.json"))
  (write-char #\Newline))
