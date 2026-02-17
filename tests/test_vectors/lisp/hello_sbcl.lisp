;; SBCL script: writes a deterministic file to the current working directory.
(with-open-file (s "out.txt"
                   :direction :output
                   :if-exists :supersede
                   :if-does-not-exist :create
                   :external-format :utf-8)
  (write-string "hello
" s))
