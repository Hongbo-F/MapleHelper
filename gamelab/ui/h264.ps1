# Re-encode a video to H.264 MP4 with Windows' own Media Foundation encoder (WinRT MediaTranscoder), for players that
# don't take OpenCV's MPEG-4 Part 2 ('mp4v'): the web UI's review videos (gamelab/ui/video.py). No download, nothing
# installed: Windows 10's built-in codecs.
#   powershell -NoProfile -ExecutionPolicy Bypass -File h264.ps1 -In X.mp4 -Out Y.mp4 [-Width 1696 -Height 1030 -Fps 15 -Bitrate 12000000]
param(
    [Parameter(Mandatory = $true)][string]$In,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Width = 1696,
    [int]$Height = 1030,
    [int]$Fps = 15,
    [int]$Bitrate = 12000000
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Transcoding.MediaTranscoder, Windows.Media.Transcoding, ContentType = WindowsRuntime]
$null = [Windows.Media.MediaProperties.MediaEncodingProfile, Windows.Media.MediaProperties, ContentType = WindowsRuntime]

$ext = [System.WindowsRuntimeSystemExtensions].GetMethods()
$asTaskOp = $ext | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
$asTaskProg = $ext | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncActionWithProgress`1' } | Select-Object -First 1

function Await($op, [Type]$type) {
    $t = $asTaskOp.MakeGenericMethod($type).Invoke($null, @($op))
    $t.Wait(-1) | Out-Null
    $t.Result
}

$inFull = [System.IO.Path]::GetFullPath($In)
$outFull = [System.IO.Path]::GetFullPath($Out)
$src = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($inFull)) ([Windows.Storage.StorageFile])
$folder = Await ([Windows.Storage.StorageFolder]::GetFolderFromPathAsync([System.IO.Path]::GetDirectoryName($outFull))) ([Windows.Storage.StorageFolder])
$dst = Await ($folder.CreateFileAsync([System.IO.Path]::GetFileName($outFull), [Windows.Storage.CreationCollisionOption]::ReplaceExisting)) ([Windows.Storage.StorageFile])

$profile = [Windows.Media.MediaProperties.MediaEncodingProfile]::CreateMp4([Windows.Media.MediaProperties.VideoEncodingQuality]::HD1080p)
$profile.Audio = $null
$profile.Video.Width = $Width
$profile.Video.Height = $Height
$profile.Video.Bitrate = $Bitrate
$profile.Video.FrameRate.Numerator = $Fps
$profile.Video.FrameRate.Denominator = 1

$tr = New-Object Windows.Media.Transcoding.MediaTranscoder
$tr.HardwareAccelerationEnabled = $false
$prep = Await ($tr.PrepareFileTranscodeAsync($src, $dst, $profile)) ([Windows.Media.Transcoding.PrepareTranscodeResult])
if (-not $prep.CanTranscode) {
    Write-Output "cannot transcode: $($prep.FailureReason)"
    exit 2
}
$t = $asTaskProg.MakeGenericMethod([double]).Invoke($null, @($prep.TranscodeAsync()))
$t.Wait(-1) | Out-Null
Write-Output "h264: $outFull"
