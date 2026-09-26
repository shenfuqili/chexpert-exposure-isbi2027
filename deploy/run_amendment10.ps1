# Amendment 10 (post hoc, exploratory): matched-exposure crossover on the final snapshots. CPU only.
# Writes results\amendment10\matched.json and results\amendment10\DONE (exit code).
$py = "D:\bench_tmp\venv\Scripts\python.exe"
$root = "D:\isbi"
$env:PYTHONIOENCODING = "utf-8"
$out = "$root\results\amendment10"
New-Item -ItemType Directory -Force -Path $out, "$root\logs" | Out-Null
Remove-Item "$out\DONE" -ErrorAction SilentlyContinue
$argv = @("$root\scripts\amendment10_matched.py", "--run1", "$root\runs\run1", "--run2", "$root\runs\run2",
          "--groups", "$root\data\groups.csv", "--out", $out, "--jobs", "8")
$p = Start-Process -FilePath $py -ArgumentList $argv -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput "$root\logs\a10.log" -RedirectStandardError "$root\logs\a10.err"
Set-Content -Path "$out\DONE" -Value $p.ExitCode
