@echo off
rem Runs the llama.cpp baseline in a fully separate session via Task Scheduler,
rem so it survives the launching terminal being closed or interrupted.
rem
rem Task Scheduler does NOT inherit your shell's environment, so anything the
rem harness reads from env is set explicitly here.
rem
rem Register once (PowerShell, from the repo root):
rem   $a = New-ScheduledTaskAction -Execute "$PWD\scripts\run_baseline_detached.cmd" -WorkingDirectory $PWD
rem   $s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 3) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
rem   Register-ScheduledTask -TaskName MiniLLM-Baseline -Action $a -Settings $s -Force
rem Then:  Start-ScheduledTask MiniLLM-Baseline
rem Watch: type results\baseline_gen_run.log

set ROOT=%~dp0..
cd /d "%ROOT%"
if not exist results mkdir results

set HF_HOME=D:\minillm\hf
set HF_HUB_CACHE=D:\minillm\hf\hub
set MINILLM_LLAMA_BIN=D:\minillm\llamacpp

rem --generate is the ONLY mode that works for a model larger than RAM on
rem llama.cpp b10437: llama-bench always repacks and cannot allocate. The
rem harness passes --no-repack --perf and closes stdin so llama-cli exits.
python bench\llama_baseline.py --generate --threads 4 --n-gen 12 --ctx 512 --load-mode mmap --timeout 7200 > results\baseline_gen_run.log 2>&1
