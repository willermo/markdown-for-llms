-- The references of document.md, as pandoc's commonmark_x reader reads them (a custom writer:
-- `pandoc -f commonmark_x -t references.lua`). One flat JSON object:
--   targets: the destination of every image and link, nested ones included (an image in the
--            alternative text of an image is a child of it, and the walk reaches it);
--   html:    the text of every raw HTML element, whose img src and a href the engine reads.
-- Code and math are not walked as references: their text is text.

function Writer(doc, opts)
  local targets, html = {}, {}
  doc:walk({
    Image = function(el) targets[#targets + 1] = el.src end,
    Link = function(el) targets[#targets + 1] = el.target end,
    RawInline = function(el) if el.format == "html" then html[#html + 1] = el.text end end,
    RawBlock = function(el) if el.format == "html" then html[#html + 1] = el.text end end,
  })
  return pandoc.json.encode({targets = targets, html = html})
end
