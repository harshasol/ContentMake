# ContentBlitz

A conversational assistant that researches a subject and turns it into marketing content (SEO blogs, LinkedIn posts, images) in a consistent brand voice.

## Language

### Research

**Topic**:
The subject the user wants content about; it owns one Research Brief and groups every Content Piece built from it.
_Avoid_: Subject, theme, campaign

**Research Brief**:
The structured, source-cited summary of research on a subject, shown to the user as the research report and used as input by every writer.
_Avoid_: Research report, research results, findings doc

**Research Record**:
The raw research gathered on a Topic (the queries run, the providers that answered, when it was gathered, and the ranked Sources) before the Content Strategist synthesises it into a Research Brief.
_Avoid_: Research results, research data

**Source**:
A web page or document gathered by research and cited in a Research Brief, with title, URL, site and retrieval date. Its number is its rank in the Research Record, and is the citation number everywhere it is cited.
_Avoid_: Reference, citation, link

### Content

**Content Piece**:
One generated deliverable of a single format: an SEO Blog, a LinkedIn Post, or an Image.
_Avoid_: Output, asset, artifact, post (when meaning any format)

**Revision**:
One saved version of a Content Piece's text, created by one of four origins: a writer's draft, a chat refinement, a hand edit, or a *restore* (a new Revision copying an older one, content and Quality Score, and naming it; earlier Revisions are never changed). A hand edit re-runs only the rule checks, and the AI reviewer's earlier scores are shown as "not re-reviewed"; a restore copies the older Revision's content and Quality Score as they were. Refinements and hand edits always start from the latest Revision. A Revision a writer makes also carries its Quality Score.
_Avoid_: Version, draft, edit

**Attempt**:
One draft made inside a writer's automatic quality loop. Attempts are not Revisions: only the one the loop keeps is saved as a Revision.
_Avoid_: Revision, iteration

**Quality Score**:
The rating a text Content Piece (an SEO Blog or a LinkedIn Post) receives against its format's rules and its Brand Profile, together with the suggestions for improving it. Images have no Quality Score.
_Avoid_: Grade, rating, eval score

### Brand

**Brand Profile**:
The user-defined description of a brand's name, audience, tone, words to avoid, sample content and visual style, which every Content Piece must follow. A user may keep several; exactly one is active per conversation, chosen in the sidebar, and each Revision records the one it was made with. A built-in, read-only **Default** profile always exists, and a new conversation starts with it.
_Avoid_: Brand voice settings, persona, style guide

### Usage

**Run**:
One workflow invocation for one chat message or command (such as *refresh research*). It has a run id, and every paid call it makes is tagged with it, so its cost and tokens can be read back from the usage ledger.
_Avoid_: Job, session, request

**Usage ledger**:
The append-only record of every paid call (model completions, images, searches): which run, conversation and agent made it, the model or provider that answered, tokens, cost, latency, and whether it was cached or a fallback. The Usage tab and the per-run caption both read it.
_Avoid_: Usage log, cost tracker, billing
