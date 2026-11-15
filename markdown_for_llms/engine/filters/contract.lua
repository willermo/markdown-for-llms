-- Contract filter of the engine (MFL032) for pandoc 3.14, commonmark_x writer.
-- Run as:  pandoc --sandbox ... -t commonmark_x --extract-media=assets --lua-filter=/in/contract.lua
--
-- It changes the AST, never the text of the output, and records every change:
--  1. media: the mediabag is emptied and refilled under bare `asset-NNNN.<ext>` keys (identical
--     bytes share one name), so pandoc writes assets/asset-NNNN.<ext> and rewrites the image
--     sources; `data:` images left by a reader are decoded by pandoc itself (no IO); with the
--     package audit, a source naming a member of the package (also percent-encoded, as EPUB
--     references are) finds its bytes by the member's SHA-1. At the end only the media that the
--     document references stay in the mediabag: an image removed with its element is not
--     written, so the engine's audit sees it lost;
--  2. images whose source is not extracted: http(s) stays as a link, never fetched
--     (`media_external`, warning); every other source is replaced by its alternative text
--     (`image_unlinked`, `media_missing`, incomplete);
--  3. links: kept for `#...`, http(s), mailto and extracted media; any other target is unlinked,
--     the text kept (`link_unlinked`). A kept destination must pass the same checks as the
--     product's reference profile (`references.inspect`, PA4): strict percent-decoding to UTF-8,
--     no backslash or control character, a well-formed authority; otherwise it is unlinked;
--  4. attributes: in every Attr the resource attributes, `style` and the `on*` handlers are
--     removed (any case), table widths and alignments reset, so the writer emits no `style`;
--  5. raw HTML (PA5): tags are read by a small tokenizer aware of quotes, case and unquoted
--     values; active and embedding elements (`script`, `style`, `iframe`, `svg`, forms ...) are
--     removed with their content, also when pandoc split them over several nodes of a list
--     (inside the raw text elements, `script` and `style`, only their closing tag ends them);
--     `picture` is unwrapped (its `source` removed, its fallback `img` kept); the other tags
--     keep only safe attributes, each once; an incomplete tag, a comment or a
--     declaration is removed. An `svg` that only wraps one image (the EPUB cover idiom) becomes
--     that image when the mediabag holds it.
-- Before any change the document is written to /work/pre-filter.md (the original extraction),
-- and the report of the changes to /work/contract-report.json: the corrections and their counts,
-- the problems and, apart, their counts per code taken before the list is truncated (at most
-- LIMIT items per severity). Optional inputs: /in/package-media.json (the package audit) and,
-- for xlsx, /in/media/, whose images are placed after the table of their sheet.

local report = {media = {}, corrections = {}, problems = {}, counts = {}, problem_counts = {}}
local LIMIT = 2000
local EXT = {svm = "svm", png = "png", jpg = "jpg", jpeg = "jpg", gif = "gif", bmp = "bmp", tif = "tiff",
             tiff = "tiff", emf = "emf", wmf = "wmf", svg = "svg", webp = "webp", eps = "eps"}
local MIME = {["image/png"] = "png", ["image/jpeg"] = "jpg", ["image/gif"] = "gif", ["image/bmp"] = "bmp",
              ["image/tiff"] = "tiff", ["image/x-emf"] = "emf", ["image/emf"] = "emf", ["image/x-wmf"] = "wmf",
              ["image/wmf"] = "wmf", ["image/svg+xml"] = "svg", ["image/webp"] = "webp"}
local MIME_OF = {png = "image/png", jpg = "image/jpeg", gif = "image/gif", bmp = "image/bmp", tiff = "image/tiff",
                 emf = "image/x-emf", wmf = "image/x-wmf", svg = "image/svg+xml", webp = "image/webp"}
-- attributes never kept, in an Attr or a raw HTML tag (compared in lower case)
local DROP = {}
for _, k in ipairs({"style", "src", "href", "srcset", "poster", "data", "background", "action", "formaction",
                    "xlink:href", "ping", "code", "codebase", "archive", "cite", "longdesc", "usemap", "dynsrc",
                    "lowsrc", "profile", "manifest", "classid", "srcdoc", "lang:href"}) do DROP[k] = true end
-- elements removed with their content; VOID ones have no content
local ACTIVE = {}
for _, k in ipairs({"script", "style", "iframe", "object", "embed", "link", "meta", "base", "form", "input",
                    "button", "select", "textarea", "option", "optgroup", "frame", "frameset", "applet", "svg",
                    "template", "noscript", "audio", "video", "source", "track", "portal", "foreignobject",
                    "canvas", "dialog", "param", "noembed", "noframes", "xmp", "plaintext", "head", "title",
                    "map", "area"}) do ACTIVE[k] = true end
local VOID = {link = true, meta = true, base = true, input = true, embed = true, source = true, track = true,
              param = true, frame = true, area = true}
-- elements whose content is text up to their closing tag (no tag starts inside)
local RAWTEXT = {script = true, style = true, textarea = true, title = true, xmp = true, iframe = true,
                 noembed = true, noframes = true}
-- tags dropped, their content kept
local UNWRAP = {html = true, body = true, picture = true}

local map, n, by_sha, known, used_images = {}, 0, {}, {}, {}

local function count(k) report.counts[k] = (report.counts[k] or 0) + 1 end
local function clip(s) s = tostring(s or "") return #s > 300 and s:sub(1, 300) or s end
local function correction(kind, original)
  count(kind)
  if #report.corrections < LIMIT then table.insert(report.corrections, {kind = kind, original = clip(original)}) end
end
local listed = {}
local function problem(code, severity, path)
  report.problem_counts[code] = (report.problem_counts[code] or 0) + 1
  if (listed[severity] or 0) < LIMIT then
    listed[severity] = (listed[severity] or 0) + 1
    table.insert(report.problems, {code = code, severity = severity, path = clip(path)})
  end
end

-- 1. media ------------------------------------------------------------------------------------

local function ext_of(path, mime)
  local e = (path:match("%.([%w]+)$") or ""):lower()
  return EXT[e] or MIME[(mime or ""):lower()] or "bin"
end

local function add_media(original, mime, contents)
  local digest = pandoc.utils.sha1(contents)
  if by_sha[digest] then map[original] = by_sha[digest]; return by_sha[digest] end
  n = n + 1
  local new = string.format("asset-%04d.%s", n, ext_of(original, mime))
  map[original] = new
  by_sha[digest] = new
  known[new] = true
  mime = mime or MIME_OF[ext_of(original, nil)] or "application/octet-stream"
  pandoc.mediabag.insert(new, mime, contents)
  table.insert(report.media, {original = clip(original), path = "assets/" .. new, mime = mime,
                              bytes = #contents, sha1 = digest})
  return new
end

local function load_data(src)
  if src:sub(1, 5):lower() ~= "data:" then return nil end
  local ok, mime, contents = pcall(pandoc.mediabag.fetch, src)
  if not ok or not contents then return nil end
  correction("data_uri_extracted", src:sub(1, 40))
  return add_media("data." .. (MIME[(mime or ""):lower()] or "bin"), mime, contents)
end

-- 3. destinations (PA4) ------------------------------------------------------------------------

local function has_control(s)
  for i = 1, #s do
    local b = s:byte(i)
    if b < 32 or b == 127 then return true end
  end
  return false
end

-- The percent-decoded destination, or nil when it does not decode strictly to UTF-8.
local function percent_decode(s)
  local out, i = {}, 1
  while i <= #s do
    local c = s:sub(i, i)
    if c == "%" then
      local hex = s:sub(i + 1, i + 2)
      if not hex:match("^%x%x$") then return nil end
      table.insert(out, string.char(tonumber(hex, 16)))
      i = i + 3
    else
      table.insert(out, c)
      i = i + 1
    end
  end
  local decoded = table.concat(out)
  if not utf8.len(decoded) then return nil end
  return decoded
end

local function scheme(t) return (t:match("^(%a[%w+.-]*):") or ""):lower() end

-- "anchor", "external" (http, https), "mailto", "local" or "unsafe" for a destination as the
-- product checks it.
local function destination_kind(t)
  if t:find("\\", 1, true) or has_control(t) then return "unsafe" end
  local decoded = percent_decode(t)
  if not decoded or decoded:find("\\", 1, true) or has_control(decoded) then return "unsafe" end
  if decoded == "" or decoded:sub(1, 1) == "#" then return "anchor" end
  if decoded:sub(1, 2) == "//" then return "unsafe" end
  local s = scheme(decoded)
  if #s == 1 then return "unsafe" end                      -- a drive letter, never a scheme
  if s == "http" or s == "https" then
    local authority = decoded:match("^%a[%w+.-]*://([^/?#]*)")
    if not authority or authority == "" then return "unsafe" end
    -- brackets (IPv6 literals) and non-ASCII hosts are not checked here: unlinked, never invalid
    if authority:find("[%[%]]") or authority:find("[\128-\255]") then return "unsafe" end
    return "external"
  end
  if s == "mailto" then
    -- no authority in a mailto URI (RFC 6068); the product's profile would parse one and refuse it
    if decoded:sub(8, 9) == "//" then return "unsafe" end
    return "mailto"
  end
  if s ~= "" then return "unsafe" end
  return "local"
end

-- 2. images, 3. links ----------------------------------------------------------------------------

-- The mediabag key named by a relative reference from inside the package (by its trailing path).
local function media_key(ref)
  if not ref or destination_kind(ref) ~= "local" then return nil end
  local path = percent_decode(ref) or ref
  path = path:gsub("^%./", "")
  while path:sub(1, 3) == "../" do path = path:sub(4) end
  path = path:gsub("^%./", "")
  local found
  for original, new in pairs(map) do
    if original == path or original:sub(-(#path + 1)) == "/" .. path then
      if found and found ~= new then return nil end
      found = new
    end
  end
  return found
end

local function fix_image(el)
  el.attr = pandoc.Attr()
  if known[el.src] then return el end
  local key = map[el.src] or media_key(el.src) or load_data(el.src)
  if key then el.src = key; return el end
  local k = destination_kind(el.src)
  if k == "external" then
    problem("media_external", "warning", el.src); return el
  end
  correction("image_unlinked", el.src)
  problem("media_missing", "incomplete", el.src)
  return el.caption
end

local function fix_link(el)
  el.attr = pandoc.Attr()
  local t = el.target or ""
  if map[t] then el.target = "assets/" .. map[t]; return el end
  if known[t] then el.target = "assets/" .. t; return el end
  local k = destination_kind(t)
  if k == "anchor" or k == "external" or k == "mailto" then return el end
  correction("link_unlinked", t)
  return el.content
end

-- 4. attributes ------------------------------------------------------------------------------

local function clean_attr(attr)
  if not attr then return attr end
  local keep, removed = {}, false
  for _, kv in ipairs(attr.attributes) do
    local k = kv[1]:lower()
    if DROP[k] or k:match("^on") or k:find("href", 1, true) then removed = true
    else table.insert(keep, kv) end
  end
  if removed then
    count("attr_removed")
    return pandoc.Attr(attr.identifier, attr.classes, keep)
  end
  return attr
end

local function fix_table(el)
  for i, _ in ipairs(el.colspecs) do el.colspecs[i] = {pandoc.AlignDefault, pandoc.ColWidthDefault} end
  el.attr = clean_attr(el.attr)
  local function rows(list)
    for _, row in ipairs(list or {}) do
      row.attr = clean_attr(row.attr)
      for _, cell in ipairs(row.cells) do cell.alignment = pandoc.AlignDefault; cell.attr = clean_attr(cell.attr) end
    end
  end
  if el.head then el.head.attr = clean_attr(el.head.attr); rows(el.head.rows) end
  for _, b in ipairs(el.bodies) do b.attr = clean_attr(b.attr); rows(b.head); rows(b.body) end
  if el.foot then el.foot.attr = clean_attr(el.foot.attr); rows(el.foot.rows) end
  return el
end

local function clean_any(el) if el.attr then el.attr = clean_attr(el.attr) end return el end

-- 5. raw HTML (PA5) --------------------------------------------------------------------------

-- Tokens of an HTML fragment: {t="text"|"start"|"end"|"comment"|"other"|"bad", raw, name, attrs, selfclose}.
-- "bad" is an incomplete tag: the rest of the fragment is not read.
local function parse_tag(s, i)
  local j = i + 1
  local closing = false
  if s:sub(j, j) == "/" then closing = true; j = j + 1 end
  local name = s:match("^%a[^%s/>]*", j)
  if not name then return nil end
  j = j + #name
  local attrs = {}
  while true do
    local ws = s:match("^[%s/]*", j)
    j = j + #ws
    local c = s:sub(j, j)
    if c == "" then return nil end
    if c == ">" then
      local tok = {t = closing and "end" or "start", raw = s:sub(i, j), name = name:lower(), attrs = attrs,
                   selfclose = ws:find("/") ~= nil}
      return tok, j + 1
    end
    local an = s:match("^[^%s/>=]+", j)
    if not an then return nil end                         -- "=" where a name should start
    j = j + #an
    local value
    local w2 = s:match("^%s*", j)
    if s:sub(j + #w2, j + #w2) == "=" then
      j = j + #w2 + 1
      j = j + #s:match("^%s*", j)
      local q = s:sub(j, j)
      if q == '"' or q == "'" then
        local e = s:find(q, j + 1, true)
        if not e then return nil end
        value = s:sub(j + 1, e - 1)
        j = e + 1
      else
        value = s:match("^[^%s>]*", j)
        j = j + #value
      end
    end
    table.insert(attrs, {name = an, value = value})
  end
end

-- Position of the closing tag of the raw text element `name` at or after `from`, or nil.
local function raw_close(s, lower, name, from)
  local pat = "</" .. name
  while true do
    local a = lower:find(pat, from, true)
    if not a then return nil end
    local after = s:sub(a + #pat, a + #pat)
    if after == "" or after:match("[%s/>]") then return a end
    from = a + 1
  end
end

-- `pending`: a raw text element left open by the previous node of the list.
local function tokenize(s, pending)
  local toks, i, text_start = {}, 1, 1
  local lower
  local function flush(upto)
    if upto >= text_start then table.insert(toks, {t = "text", raw = s:sub(text_start, upto)}) end
  end
  -- the content of a raw text element is text up to its closing tag; nil when it is not closed here
  local function raw_text(name, from)
    lower = lower or s:lower()
    local a = raw_close(s, lower, name, from)
    if not a then
      if from <= #s then table.insert(toks, {t = "text", raw = s:sub(from)}) end
      return nil
    end
    if a > from then table.insert(toks, {t = "text", raw = s:sub(from, a - 1)}) end
    return a
  end
  if pending then
    local a = raw_text(pending, 1)
    if not a then return toks end
    i, text_start = a, a
  end
  while true do
    local lt = s:find("<", i, true)
    if not lt then break end
    local nx = s:sub(lt + 1, lt + 1)
    if s:sub(lt, lt + 3) == "<!--" then
      flush(lt - 1)
      local e = s:find("-->", lt + 4, true)
      if not e then table.insert(toks, {t = "bad", raw = s:sub(lt)}); return toks end
      table.insert(toks, {t = "comment", raw = s:sub(lt, e + 2)})
      i = e + 3; text_start = i
    elseif nx == "!" or nx == "?" then
      flush(lt - 1)
      local close = s:sub(lt, lt + 8) == "<![CDATA[" and "]]>" or ">"
      local e = s:find(close, lt + 2, true)
      if not e then table.insert(toks, {t = "bad", raw = s:sub(lt)}); return toks end
      e = e + #close - 1
      table.insert(toks, {t = "other", raw = s:sub(lt, e)})
      i = e + 1; text_start = i
    elseif nx:match("%a") or (nx == "/" and s:sub(lt + 2, lt + 2):match("%a")) then
      flush(lt - 1)
      local tok, nexti = parse_tag(s, lt)
      if not tok then table.insert(toks, {t = "bad", raw = s:sub(lt)}); return toks end
      table.insert(toks, tok)
      i = nexti; text_start = i
      if tok.t == "start" and RAWTEXT[tok.name] and not tok.selfclose then
        local a = raw_text(tok.name, nexti)
        if not a then return toks end
        i, text_start = a, a
      end
    else
      i = lt + 1
    end
  end
  flush(#s)
  return toks
end

local function safe_name(name) return name:match("^[%a_:][%w_.:-]*$") ~= nil end

local function serialize(tok)
  if tok.t == "end" then return "</" .. tok.name .. ">" end
  local out, seen, changed = {"<" .. tok.name}, {}, false
  for _, a in ipairs(tok.attrs) do
    local k = a.name:lower()
    if seen[k] or not safe_name(a.name) or DROP[k] or k:match("^on") or k:find("href", 1, true) then
      changed = true
    else
      seen[k] = true
      if a.value == nil then table.insert(out, " " .. k)
      else table.insert(out, " " .. k .. '="' .. a.value:gsub('"', "&quot;"):gsub("<", "&lt;"):gsub(">", "&gt;") .. '"') end
      if k ~= a.name then changed = true end
    end
  end
  table.insert(out, tok.selfclose and " />" or ">")
  return table.concat(out), changed
end

-- The one image an `svg` wraps (EPUB cover pages), from the tokens of the svg element.
local function svg_image(toks, from, to)
  local href, others = nil, 0
  for k = from + 1, to - 1 do
    local tok = toks[k]
    if tok.t == "start" then
      if tok.name == "image" or tok.name == "img" then
        if href then return nil end
        for _, a in ipairs(tok.attrs) do
          local an = a.name:lower()
          if an == "xlink:href" or an == "href" or an == "src" then href = a.value end
        end
      elseif tok.name ~= "title" and tok.name ~= "desc" and tok.name ~= "g" then
        others = others + 1
      end
    elseif tok.t == "text" and tok.raw:match("%S") then
      others = others + 1
    end
  end
  if others == 0 then return href end
  return nil
end

-- Clean one raw HTML fragment; `state` carries an active element open across nodes of a list.
-- Returns the cleaned text and, when an svg wrapped one known image, its mediabag key.
local function clean_html(text, state)
  local toks = tokenize(text, state.suppress and RAWTEXT[state.suppress] and state.depth == 1 and state.suppress or nil)
  local out, images = {}, {}
  local k = 1
  while k <= #toks do
    local tok = toks[k]
    if state.suppress then
      if tok.t == "start" and tok.name == state.suppress and not tok.selfclose then
        state.depth = state.depth + 1
      elseif tok.t == "end" and tok.name == state.suppress then
        state.depth = state.depth - 1
        if state.depth == 0 then
          if state.svg_from then
            local key = media_key(svg_image(state.svg_toks, 0, #state.svg_toks + 1))
            if key and used_images[key] then correction("svg_image_already_present", key)
            elseif key then table.insert(images, key); used_images[key] = true; correction("svg_image_recovered", key)
            else problem("object_not_converted", "incomplete", "svg") end
          end
          state.suppress, state.svg_from, state.svg_toks = nil, nil, nil
        end
      end
      if state.svg_toks then table.insert(state.svg_toks, tok) end
      count("raw_active_content_removed")
    elseif tok.t == "text" then
      table.insert(out, tok.raw)
    elseif tok.t == "start" and ACTIVE[tok.name] then
      correction("raw_active_removed", tok.raw)
      if not (VOID[tok.name] or tok.selfclose) then
        state.suppress, state.depth = tok.name, 1
        if tok.name == "svg" then state.svg_from, state.svg_toks = true, {} end
      end
    elseif tok.t == "end" and ACTIVE[tok.name] then
      correction("raw_active_removed", tok.raw)
    elseif tok.t == "start" or tok.t == "end" then
      if UNWRAP[tok.name] then
        correction("raw_tag_removed", tok.raw)
      else
        local cleaned, changed = serialize(tok)
        if changed then correction("raw_attr_removed", tok.raw) end
        table.insert(out, cleaned)
      end
    elseif tok.t == "bad" then
      correction("raw_html_rejected", tok.raw)
    else
      correction("raw_markup_removed", tok.raw)
    end
    k = k + 1
  end
  return table.concat(out), images
end

local function is_html(el) return el.format:match("^html") ~= nil end

local function clean_list(list, is_block)
  local state = {suppress = nil, depth = 0}
  local result = {}
  local changed = false
  for _, el in ipairs(list) do
    if (el.t == "RawInline" or el.t == "RawBlock") and is_html(el) then
      local text, images = clean_html(el.text, state)
      changed = true
      for _, key in ipairs(images) do
        local img = pandoc.Image({}, key)
        table.insert(result, is_block and pandoc.Para({img}) or img)
      end
      if text:match("%S") then
        table.insert(result, is_block and pandoc.RawBlock(el.format, text) or pandoc.RawInline(el.format, text))
      end
    elseif state.suppress then
      changed = true
      count("raw_active_content_removed")
      if state.svg_toks then          -- the parts of an svg pandoc split over several nodes
        el:walk({RawInline = function(r) if is_html(r) then for _, tk in ipairs(tokenize(r.text)) do
                                           table.insert(state.svg_toks, tk) end end end,
                 RawBlock = function(r) if is_html(r) then for _, tk in ipairs(tokenize(r.text)) do
                                          table.insert(state.svg_toks, tk) end end end})
      end
    else
      table.insert(result, el)
    end
  end
  if state.suppress then
    correction("raw_active_unclosed", state.suppress)
    if state.svg_from then problem("object_not_converted", "incomplete", "svg") end
  end
  if not changed then return nil end
  return is_block and pandoc.Blocks(result) or pandoc.Inlines(result)
end

-- sheets of xlsx -----------------------------------------------------------------------------

local function read_audit()
  local f = io.open("/in/package-media.json", "rb")
  if not f then return nil end
  local audit = pandoc.json.decode(f:read("a")); f:close()
  return audit
end

-- The members of the package named by their bytes: a source that names a member finds its key.
local function map_members(audit)
  for _, m in ipairs(audit.media or {}) do
    if m.sha1 and by_sha[m.sha1] and not map[m.member] then map[m.member] = by_sha[m.sha1] end
  end
end

local function place_sheet_images(doc, audit)
  if not audit or not audit.sheets or #audit.sheets == 0 then return doc end
  local by_index = {}
  for _, s in ipairs(audit.sheets or {}) do by_index[s.index] = s end
  local blocks, sheet = pandoc.Blocks{}, nil
  local function flush()
    local s = sheet and by_index[sheet]
    if not s then return end
    local placed = {}
    for _, img in ipairs(s.images or {}) do
      local seen = placed[img.sha1 .. img.cell]
      placed[img.sha1 .. img.cell] = true
      local h = (not seen) and io.open("/in/media/" .. img.file, "rb")
      if h then
        local bytes = h:read("a"); h:close()
        local key = add_media(img.member, nil, bytes)
        local alt = (img.descr ~= "" and img.descr) or img.name or ""
        blocks:insert(pandoc.Para{pandoc.Image({pandoc.Str(alt)}, key, s.name .. "!" .. img.cell)})
        correction("sheet_image_placed", img.member)
      end
    end
  end
  for _, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.identifier:match("^sheet%-%d+$") then
      flush(); sheet = tonumber(b.identifier:match("%d+"))
    end
    blocks:insert(b)
  end
  flush()
  doc.blocks = blocks
  return doc
end

-- Only the media the body references stay in the mediabag (and in the report).
local function prune_media(doc)
  local referenced = {}
  pandoc.Blocks(doc.blocks):walk({
    Image = function(el) if known[el.src] then referenced[el.src] = true end end,
    Link = function(el)
      local t = el.target or ""
      if t:sub(1, 7) == "assets/" and known[t:sub(8)] then referenced[t:sub(8)] = true end
    end})
  local kept = {}
  for _, m in ipairs(report.media) do
    local key = m.path:sub(8)
    if referenced[key] then table.insert(kept, m)
    else pandoc.mediabag.delete(key); count("media_unreferenced_dropped") end
  end
  report.media = kept
end

function Pandoc(doc)
  local pre = io.open("/work/pre-filter.md", "wb")
  pre:write(pandoc.write(doc, "commonmark_x", {wrap_text = "wrap-none"})); pre:close()
  local items = {}
  for fp, mt, contents in pandoc.mediabag.items() do table.insert(items, {fp, mt, contents}) end
  pandoc.mediabag.empty()
  for _, it in ipairs(items) do add_media(it[1], it[2], it[3]) end
  local audit = read_audit()
  if audit then map_members(audit) end
  doc = place_sheet_images(doc, audit)
  doc:walk({Image = function(el) local key = map[el.src] or (known[el.src] and el.src)
                                  if key then used_images[key] = true end end})
  -- top-down: a list is cleaned before the lists inside its elements, so an active element
  -- removed here is never half-cleaned below
  doc = doc:walk({traverse = "topdown",
                  Inlines = function(l) return clean_list(l, false) end,
                  Blocks = function(l) return clean_list(l, true) end})
  doc = doc:walk({Image = fix_image, Link = fix_link, Table = fix_table,
                  Span = clean_any, Div = clean_any, Header = clean_any, CodeBlock = clean_any,
                  Code = clean_any, Figure = clean_any})
  prune_media(doc)
  local w = io.open("/work/contract-report.json", "w")
  w:write(pandoc.json.encode(report)); w:close()
  return doc
end
