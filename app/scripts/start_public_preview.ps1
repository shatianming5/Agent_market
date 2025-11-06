param(
  [switch]$StartBackend,
  [switch]$StartFrontend
)

$ErrorActionPreference = 'SilentlyContinue'

function Ensure-Cloudflared {
  $dir = Join-Path "user_data" "tmp"
  if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $cf = Join-Path $dir "cloudflared.exe"
  if (!(Test-Path $cf)) {
    Write-Host "Downloading cloudflared ..."
    $url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
    Invoke-WebRequest -Uri $url -OutFile $cf -UseBasicParsing
  }
  return $cf
}

function Test-Listen($port){
  try {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $conn
  } catch { return $false }
}

function Start-LocalBackend {
  if (Test-Listen 8032) { return }
  Write-Host "Starting backend (uvicorn) on 8032 ..."
  $root = (Resolve-Path ".").Path
  Start-Process -FilePath "conda" -ArgumentList @("run","--no-capture-output","-n","freqtrade","uvicorn","server.main:app","--host","127.0.0.1","--port","8032") -WorkingDirectory $root -WindowStyle Minimized | Out-Null
  Start-Sleep -Seconds 3
}

function Start-LocalFrontend {
  if (Test-Listen 5173) { return }
  if (!(Test-Path "web")) { return }
  Write-Host "Starting frontend (vite) on 5173 ..."
  Push-Location web
  Start-Process -FilePath cmd.exe -ArgumentList @("/c","npm","run","dev","--silent") -WindowStyle Minimized | Out-Null
  Pop-Location
  Start-Sleep -Seconds 3
}

function Start-Tunnel($localUrl, $name){
  $cf = Ensure-Cloudflared
  $log = Join-Path "user_data/tmp" ("cf_" + $name + ".log")
  if (Test-Path $log) { Remove-Item $log -Force }
  $args = @('tunnel','--url', $localUrl)
  $p = Start-Process -FilePath $cf -ArgumentList $args -RedirectStandardOutput $log -WindowStyle Minimized -PassThru
  $tries = 0
  $url = $null
  while ($tries -lt 15 -and -not $url) {
    Start-Sleep -Seconds 1
    if (Test-Path $log) {
      $match = Select-String -Path $log -Pattern 'https://[\w\.-]+\.trycloudflare\.com' | Select-Object -Last 1
      if ($match) { $url = $match.Matches[0].Value }
    }
    $tries++
  }
  [PSCustomObject]@{ Url = $url; Pid = $p.Id; Log = $log }
}

if ($StartBackend) { Start-LocalBackend }
if ($StartFrontend) { Start-LocalFrontend }

Write-Host "Starting Cloudflare quick tunnels ..."

$backend = Start-Tunnel "http://127.0.0.1:8032" "api8032"
if (-not $backend.Url) { Write-Warning "Backend tunnel URL not detected. Check $($backend.Log)" }
else { Write-Host ("Backend public URL: " + $backend.Url) }

if (Test-Listen 5173) {
  $frontend = Start-Tunnel "http://127.0.0.1:5173" "web5173"
  if (-not $frontend.Url) { Write-Warning "Frontend tunnel URL not detected. Check $($frontend.Log)" }
  else { Write-Host ("Frontend public URL: " + $frontend.Url) }
  if ($backend.Url) {
    Write-Host "Tip: Start frontend with backend URL:"
    Write-Host ("  set VITE_API_BASE=" + $backend.Url)
    Write-Host "  npm run dev"
  }
} else {
  Write-Warning "Frontend (5173) not listening. Start Vite dev first, then rerun this script to expose." 
}

Write-Host "Done. (Keep this script's cloudflared processes running to maintain public access.)"

