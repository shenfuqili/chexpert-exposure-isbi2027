# Amendment 8 (post hoc): RAD-DINO head score on the 720 bridge images, unfiltered (pipeline check against the Mac's
# Amendment 6 result) and low-passed to 224 px. GPU, a few minutes each; writes results\amendment8\DONE with both
# exit codes (unfiltered, lowpass224).
$py = "D:\bench_tmp\venv\Scripts\python.exe"
$root = "D:\isbi"
$env:PYTHONIOENCODING = "utf-8"
$out = "$root\results\amendment8"
New-Item -ItemType Directory -Force -Path $out, "$root\logs" | Out-Null
Remove-Item "$out\DONE" -ErrorAction SilentlyContinue
$common = @("$root\scripts\head_score.py", "raddino", "--per-image", "$root\bridge\results\bridge\bridge_per_image.csv",
            "--head", "$root\data\rad_dino\dino_head.safetensors", "--out", $out, "--device", "cuda",
            "--root", "$root\bridge", "--model", "$root\data\rad_dino\hf")
$procs = foreach ($variant in @(@{tag = "unfiltered"; extra = @()}, @{tag = "lowpass224"; extra = @("--lowpass", "224")})) {
    $p = Start-Process -FilePath $py -ArgumentList ($common + $variant.extra) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$root\logs\a8_$($variant.tag).log" -RedirectStandardError "$root\logs\a8_$($variant.tag).err"
    $null = $p.Handle  # keeps the handle so ExitCode is available after exit
    $p
}
$procs | Wait-Process
Set-Content -Path "$out\DONE" -Value (($procs | ForEach-Object { $_.ExitCode }) -join ",")
