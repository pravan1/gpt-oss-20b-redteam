# PowerShell script to run GPT-OSS-20B Red Team tests
# Usage: .\run_redteam.ps1

Write-Host "GPT-OSS-20B Red Team Runner" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

# Check if environment variables are set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "`nPlease set your OPENAI_API_KEY environment variable:" -ForegroundColor Yellow
    Write-Host '  $env:OPENAI_API_KEY = "your-api-key-here"' -ForegroundColor Green
    Write-Host ""
    exit 1
}

if (-not $env:OPENAI_BASE_URL) {
    Write-Host "Setting default OPENAI_BASE_URL to https://api.openai.com/v1" -ForegroundColor Yellow
    $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
}

# Also set GPT_OSS_* variables for compatibility
$env:GPT_OSS_API_KEY = $env:OPENAI_API_KEY
$env:GPT_OSS_BASE_URL = $env:OPENAI_BASE_URL

Write-Host "`nEnvironment configured:" -ForegroundColor Green
Write-Host "  API Base URL: $env:OPENAI_BASE_URL" -ForegroundColor Gray
Write-Host "  API Key: [SET]" -ForegroundColor Gray

# Parse command line arguments
param(
    [string]$Backend = "openai_compatible",
    [string]$Categories = "reward_hacking,deception,hidden_motivation,eval_awareness,cot_issues",
    [string]$Seeds = "0,1,2",
    [string]$Output = "runs/realapi",
    [switch]$Mock = $false
)

if ($Mock) {
    Write-Host "`nRunning in MOCK mode for testing..." -ForegroundColor Yellow
    $Backend = "mock"
    $Output = "runs/mock-test"
}

Write-Host "`nConfiguration:" -ForegroundColor Cyan
Write-Host "  Backend: $Backend"
Write-Host "  Categories: $Categories"
Write-Host "  Seeds: $Seeds"
Write-Host "  Output: $Output"

Write-Host "`nStarting red team tests..." -ForegroundColor Green

# Run the tests
python -m gpt_oss_redteam.cli run `
    --backend $Backend `
    --categories $Categories `
    --seeds $Seeds `
    --out $Output

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTests completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: $Output" -ForegroundColor Cyan
    
    # Generate report
    Write-Host "`nGenerating report..." -ForegroundColor Yellow
    python -m gpt_oss_redteam.cli report --run-dir $Output
    
} else {
    Write-Host "`nTests failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review results in $Output" -ForegroundColor Gray
Write-Host "2. Generate findings: python -m gpt_oss_redteam.cli findings --issue-id ISSUE_1 --out findings_templates/finding_1.json" -ForegroundColor Gray
Write-Host "3. Validate findings: pytest -k test_reporting -q" -ForegroundColor Gray