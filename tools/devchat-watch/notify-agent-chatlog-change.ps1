param( 
    [ValidateSet('Codex', 'ClaudeCode')] 
    [string]$Agent = 'Codex', 
    [int]$RecentMinutes = 5, 
    [string]$ChatlogPath, 
    [string]$StatePath, 
    [switch]$ShowPopup, 
    [switch]$CopyPrompt, 
    [switch]$Force, 
    [switch]$NoState 
) 
 
Set-StrictMode -Version Latest 
$ErrorActionPreference = 'Stop' 
 
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path 
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptRoot) 
 
if (-not $ChatlogPath) { 
    $ChatlogPath = Join-Path $RepoRoot '.private\devchats\chatlog.md' 
} 
if (-not $StatePath) { 
    $StatePath = Join-Path $RepoRoot ('.private\devchats\.chatlog-notify-{0}.json' -f $Agent.ToLowerInvariant()) 
}
 
if (-not (Test-Path -LiteralPath $ChatlogPath)) { 
    throw ('chatlog.md was not found at: {0}' -f $ChatlogPath) 
} 
 
$ChatlogItem = Get-Item -LiteralPath $ChatlogPath 
$LastWriteUtc = $ChatlogItem.LastWriteTimeUtc 
$Age = [DateTime]::UtcNow - $LastWriteUtc 
 
if ($Age.TotalMinutes -gt $RecentMinutes -and -not $Force) { 
    Write-Host ('No notification sent. chatlog.md is older than {0} minutes.' -f $RecentMinutes) 
    exit 0 
} 
 
if (-not $NoState -and (Test-Path -LiteralPath $StatePath)) { 
    try { 
        $State = ConvertFrom-Json (Get-Content -LiteralPath $StatePath -Raw) 
        if ($State.lastWriteTimeUtc -eq $LastWriteUtc.ToString('o') -and -not $Force) { 
            Write-Host ('No notification sent. This version of chatlog.md was already notified for {0}.' -f $Agent) 
            exit 0 
        } 
    } catch { 
        Write-Warning 'State file exists but could not be parsed. Continuing without prior state.' 
    } 
}
 
$Title = 'Developer Chatlog Update for {0}' -f $Agent 
$Prompt = 'chatlog.md changed very recently. Please inspect it and decide whether any new entry needs a response from you.' 
$MessageLines = @( 
    $Title, 
    '', 
    $Prompt, 
    '', 
    ('Path: {0}' -f $ChatlogPath), 
    ('Last updated (UTC): {0}' -f $LastWriteUtc.ToString('u').TrimEnd()) 
) 
$Message = $MessageLines -join [Environment]::NewLine 
 
Write-Host $Message 
 
if ($CopyPrompt -and (Get-Command Set-Clipboard -ErrorAction SilentlyContinue)) { 
    Set-Clipboard -Value $Message 
    Write-Host 'Copied notification text to the clipboard.' 
} 
 
if ($ShowPopup) { 
    $WshShell = New-Object -ComObject WScript.Shell 
    [void]$WshShell.Popup($Message, 8, $Title, 64) 
}
 
if (-not $NoState) { 
    $StateObject = @{ 
        agent = $Agent 
        lastWriteTimeUtc = $LastWriteUtc.ToString('o') 
        notifiedAtUtc = [DateTime]::UtcNow.ToString('o') 
    } 
    $Json = ConvertTo-Json $StateObject 
    [System.IO.File]::WriteAllText($StatePath, $Json) 
}
