" Vim syntax file
" Language:	tutorial (simplified markup by bill)
" Maintainer:	No One
" Last Change:

" Quit when a (custom) syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

if !exists ("b:color")
   color pablo     
endif

syn match     tutorInstruction        "^====>.*"
" syn match     tutorDocumentation      ">> .*"
syn match     tutorDocumentation      "^!! .*"
syn match     tutorResult             "{.*"

hi def link tutorDocumentation    PreProc
hi def link tutorInstruction      Operator
hi def link tutorResult           String

let b:current_syntax = "tutor"
