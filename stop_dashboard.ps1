Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*streamlit*dashboard.py*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
