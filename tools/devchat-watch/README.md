# Devchat Watch 
 
This tool checks whether `.private/devchats/chatlog.md` changed very recently and, if so, emits a notification for either `Codex` or `ClaudeCode` to inspect the chatlog and decide whether a response is needed. 
 
## Files 
 
- `notify-agent-chatlog-change.ps1`: main watcher script 
- `notify-agent-chatlog-change.cmd`: Windows-friendly wrapper 
 
## Usage 
 
From the repo root or the tool folder: 
 
```cmd 
tools\devchat-watch\notify-agent-chatlog-change.cmd 
``` 
 
Target Claude Code instead: 
 
```cmd 
tools\devchat-watch\notify-agent-chatlog-change.cmd -Agent ClaudeCode 
``` 
 
Use a different freshness window in minutes: 
 
```cmd 
tools\devchat-watch\notify-agent-chatlog-change.cmd -RecentMinutes 10 
``` 
 
Show a popup and copy the notification text to the clipboard: 
 
```cmd 
tools\devchat-watch\notify-agent-chatlog-change.cmd -ShowPopup -CopyPrompt 
``` 
 
Force a notification even if the file is older or already notified: 
 
```cmd 
tools\devchat-watch\notify-agent-chatlog-change.cmd -Force 
``` 
 
## Duplicate Suppression 
 
By default the script records the last notified write timestamp in `.private/devchats/.chatlog-notify-<agent>.json` so the same chatlog version is not repeatedly announced. 
 
Use `-NoState` if you want a stateless run. 
 
## Future Design Direction 
 
This first version is intentionally small, but it should be treated as the seed of a more general watcher/dispatcher. 
Future extensions should separate (1) trigger conditions from (2) actions, so we can add more conditions and more response types without rewriting the whole tool. 
Likely future triggers include targeted-agent detection, unanswered-entry detection, mention parsing, and scheduled polling. 
Likely future actions include queue-file output, tool launch, webhook dispatch, and richer agent-specific prompts.
 
## Current Scope 
 
This version only emits a notification message for a human or agent process to notice. It does not directly invoke Codex, Claude Code, or any external API yet.
