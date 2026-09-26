# One-shot re-analysis after the 2026-09-25 fixes (partner-run membership reference; secondary endpoints 4, 5, 6).
# CPU only. Three pre-registered variants run in parallel; writes results\final_v2\DONE with their exit codes.
$py = "D:\bench_tmp\venv\Scripts\python.exe"
$root = "D:\isbi"
$env:PYTHONIOENCODING = "utf-8"
$variants = @(@("uzero", "P_base"), @("uone", "P_base"), @("uzero", "P_clean"))
Remove-Item "$root\results\final_v2\DONE" -ErrorAction SilentlyContinue
$procs = foreach ($v in $variants) {
    $tag = "$($v[0])_$($v[1])"
    $out = "$root\results\final_v2\$tag"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $argv = @("$root\scripts\analyze.py", "--groups", "$root\data\groups.csv", "--run1", "$root\runs\run1",
              "--run2", "$root\runs\run2", "--out", $out, "--policy", $v[0], "--probe-set", $v[1],
              "--n-boot", "10000", "--n-perm", "2000", "--progress", "$out\progress.json")
    $p = Start-Process -FilePath $py -ArgumentList $argv -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$root\logs\reanalysis_$tag.log" -RedirectStandardError "$root\logs\reanalysis_$tag.err"
    $null = $p.Handle  # keeps the handle so ExitCode is available after exit
    $p
}
$procs | Wait-Process
Set-Content -Path "$root\results\final_v2\DONE" -Value (($procs | ForEach-Object { $_.ExitCode }) -join ",")
