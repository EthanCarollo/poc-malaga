param([switch]$ServerOwned)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ApiRoot = 'http://127.0.0.1:7860'
$ServerProcess = $null
$ReferenceData = $null

Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Xaml

function Get-LocalHealth {
  try {
    $health = Invoke-RestMethod -Uri "$ApiRoot/api/health" -TimeoutSec 2
    if ($health.application -ne 'poc-malaga') { return $null }
    return $health
  } catch {
    return $null
  }
}

function Start-LocalApi {
  $health = Get-LocalHealth
  if ($health) { return $false }

  $wsl = Join-Path $env:SystemRoot 'System32\wsl.exe'
  if (-not (Test-Path $wsl)) { throw "WSL2 est introuvable. Installe WSL2 et configure les moteurs avant de lancer l’atelier." }
  $distributions = @(& $wsl --list --quiet 2>$null | ForEach-Object {
    ([string]$_).Replace([string][char]0, '').Trim()
  } | Where-Object { $_ -and $_ -notlike 'docker-desktop*' })
  $distribution = $distributions | Where-Object { $_ -match '^Ubuntu' } | Select-Object -First 1
  if (-not $distribution) { $distribution = $distributions | Select-Object -First 1 }
  if (-not $distribution) { throw "Aucune distribution Linux WSL n’est installée. Installe Ubuntu, puis réessaie." }

  $linuxProject = (& $wsl --distribution $distribution --exec wslpath -a $ProjectRoot 2>$null | Out-String).Trim()
  if (-not $linuxProject) { throw "Impossible de résoudre le chemin du projet dans WSL ($distribution). Vérifie que sa distribution démarre correctement." }

  $arguments = "--distribution $distribution --cd $linuxProject --exec python3 server.py"
  $logDirectory = Join-Path $ProjectRoot 'outputs'
  if (-not (Test-Path $logDirectory)) { New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null }
  $stdoutPath = Join-Path $logDirectory 'native-api.stdout.log'
  $stderrPath = Join-Path $logDirectory 'native-api.stderr.log'
  $script:ServerProcess = Start-Process -FilePath $wsl -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
  $deadline = [DateTime]::UtcNow.AddSeconds(18)
  do {
    Start-Sleep -Milliseconds 350
    $health = Get-LocalHealth
    if ($health) { return $true }
    if ($script:ServerProcess.HasExited) { break }
  } while ([DateTime]::UtcNow -lt $deadline)

  $details = @()
  if ($script:ServerProcess.HasExited) { $details += "wsl.exe a terminé avec le code $($script:ServerProcess.ExitCode)." }
  if (Test-Path $stderrPath) { $details += (Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue).Trim() }
  if (Test-Path $stdoutPath) { $details += (Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue).Trim() }
  throw "L’API locale n’a pas démarré. Vérifie Python 3 et le port 7860. $($details -join ' ')"
}

try {
  $ServerOwned = Start-LocalApi
} catch {
  [System.Windows.MessageBox]::Show($_.Exception.Message, 'POC Malaga · démarrage impossible', [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Error) | Out-Null
  exit 1
}

[xml]$xaml = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'MainWindow.xaml') -Raw -Encoding UTF8
$reader = New-Object System.Xml.XmlNodeReader $xaml
$window = [Windows.Markup.XamlReader]::Load($reader)

foreach ($name in @(
  'NavCreate','NavHistory','NavSettings','PageTitle','PageSubtitle','StatusDot','StatusLabel','GpuBadge',
  'CreatePage','HistoryPage','SettingsPage','PromptBox','CharacterCount','ModelBox','ResolutionBox',
  'UpsamplingBox','ReferenceButton','ReferenceLabel','GenerateButton','GenerationStatus','ResultImage',
  'ResultPlaceholder','ResultMeta','HistoryItems','RefreshHistoryButton','HealthDetails','RefreshHealthButton'
)) {
  Set-Variable -Name $name -Value $window.FindName($name) -Scope Script
}

$script:ReferenceData = $null
$script:HistoryCache = @()
$script:Generator = New-Object System.ComponentModel.BackgroundWorker

function Set-Page([string]$page) {
  $script:CreatePage.Visibility = 'Collapsed'
  $script:HistoryPage.Visibility = 'Collapsed'
  $script:SettingsPage.Visibility = 'Collapsed'
  $script:NavCreate.Background = 'Transparent'; $script:NavCreate.BorderBrush = 'Transparent'; $script:NavCreate.Foreground = '#A8B2C1'
  $script:NavHistory.Background = 'Transparent'; $script:NavHistory.BorderBrush = 'Transparent'; $script:NavHistory.Foreground = '#A8B2C1'
  $script:NavSettings.Background = 'Transparent'; $script:NavSettings.BorderBrush = 'Transparent'; $script:NavSettings.Foreground = '#A8B2C1'
  switch ($page) {
    'create' {
      $script:CreatePage.Visibility = 'Visible'; $script:PageTitle.Text = 'Créer une image'
      $script:PageSubtitle.Text = 'Génération privée, directement sur votre machine'
      $script:NavCreate.Background = '#233047'; $script:NavCreate.BorderBrush = '#4069CF'; $script:NavCreate.Foreground = '#F3F5F8'
    }
    'history' {
      $script:HistoryPage.Visibility = 'Visible'; $script:PageTitle.Text = 'Historique'
      $script:PageSubtitle.Text = 'Retrouvez vos images générées localement'
      $script:NavHistory.Background = '#233047'; $script:NavHistory.BorderBrush = '#4069CF'; $script:NavHistory.Foreground = '#F3F5F8'
      Update-History
    }
    'settings' {
      $script:SettingsPage.Visibility = 'Visible'; $script:PageTitle.Text = 'Réglages'
      $script:PageSubtitle.Text = 'État de la machine et des moteurs locaux'
      $script:NavSettings.Background = '#233047'; $script:NavSettings.BorderBrush = '#4069CF'; $script:NavSettings.Foreground = '#F3F5F8'
      Update-Health
    }
  }
}

function Get-ImageSource([string]$path) {
  $bitmap = New-Object System.Windows.Media.Imaging.BitmapImage
  $bitmap.BeginInit()
  $bitmap.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
  $bitmap.UriSource = [Uri]::new("$ApiRoot$path")
  $bitmap.EndInit()
  $bitmap.Freeze()
  return $bitmap
}

function Update-Health {
  $health = Get-LocalHealth
  if (-not $health) {
    $script:HealthDetails.Text = 'Le serveur local ne répond pas.'
    $script:GpuBadge.Text = 'API locale · hors ligne'
    $script:StatusLabel.Text = 'API locale indisponible'
    $script:StatusDot.Fill = '#DB6B6B'
    return
  }

  $gpuNames = @($health.gpu | ForEach-Object { "$($_.name) · $([Math]::Round($_.memory_mb / 1024, 1)) Go" })
  $gpuText = if ($gpuNames.Count) { $gpuNames -join ', ' } else { 'GPU NVIDIA non détecté par WSL' }
  $script:GpuBadge.Text = "GPU · $gpuText"
  $fluxStatus = if ($health.runtimes.flux3) { 'installé' } else { 'à configurer' }
  $qwenStatus = if ($health.runtimes.qwen21) { 'installé' } else { 'à configurer' }
  $lines = @(
    "Serveur local : prêt",
    "Matériel : $gpuText",
    "FLUX 3 Image Klein FP8 : runtime $fluxStatus",
    "Qwen-Image-2.1 BF16 : runtime $qwenStatus",
    'Un seul moteur est chargé à la fois. Le premier lancement peut prendre plusieurs minutes.'
  )
  if ($health.flux_vram_warning) { $lines += 'Attention : FLUX utilise beaucoup de mémoire GPU. Le mode FP8 séquentiel ne garantit pas une génération sur une carte de 8 Go.' }
  $script:HealthDetails.Text = $lines -join "`n"
  $script:StatusLabel.Text = 'API locale prête'
  $script:StatusDot.Fill = '#4CCB91'
}

function Update-History {
  $script:HistoryItems.Children.Clear()
  try {
    $response = Invoke-RestMethod -Uri "$ApiRoot/api/history" -TimeoutSec 5
    $script:HistoryCache = @($response.items)
  } catch {
    $script:HistoryCache = @()
    $script:HistoryItems.Children.Add((New-Object System.Windows.Controls.TextBlock -Property @{ Text = "Impossible de charger l’historique local."; Foreground = '#DB8B8B'; Margin = '8,8,8,0' })) | Out-Null
    return
  }

  if ($script:HistoryCache.Count -eq 0) {
    $script:HistoryItems.Children.Add((New-Object System.Windows.Controls.TextBlock -Property @{ Text = 'Aucune image générée pour le moment.'; Foreground = '#98A3B4'; Margin = '8,8,8,0' })) | Out-Null
    return
  }

  foreach ($entry in $script:HistoryCache) {
    $card = New-Object System.Windows.Controls.Border
    $card.Background = '#141A22'; $card.BorderBrush = '#29313D'; $card.BorderThickness = '1'; $card.CornerRadius = '10'; $card.Padding = '12'; $card.Margin = '0,0,0,10'
    $layout = New-Object System.Windows.Controls.Grid
    $imageColumn = New-Object System.Windows.Controls.ColumnDefinition
    $imageColumn.Width = [System.Windows.GridLength]::new(190)
    $detailsColumn = New-Object System.Windows.Controls.ColumnDefinition
    $detailsColumn.Width = [System.Windows.GridLength]::new(1, [System.Windows.GridUnitType]::Star)
    $buttonColumn = New-Object System.Windows.Controls.ColumnDefinition
    $buttonColumn.Width = [System.Windows.GridLength]::Auto
    $layout.ColumnDefinitions.Add($imageColumn)
    $layout.ColumnDefinitions.Add($detailsColumn)
    $layout.ColumnDefinitions.Add($buttonColumn)
    $preview = New-Object System.Windows.Controls.Image
    $preview.Width = 170; $preview.Height = 112; $preview.Stretch = 'UniformToFill'; $preview.Margin = '0,0,16,0'
    try { $preview.Source = Get-ImageSource $entry.image } catch { }
    [System.Windows.Controls.Grid]::SetColumn($preview, 0); $layout.Children.Add($preview) | Out-Null
    $details = New-Object System.Windows.Controls.StackPanel
    $model = New-Object System.Windows.Controls.TextBlock
    $model.Text = "$($entry.model_label) · $($entry.precision)"; $model.FontWeight = 'SemiBold'; $model.Foreground = '#F3F5F8'; $model.Margin = '0,4,0,7'
    $prompt = New-Object System.Windows.Controls.TextBlock
    $prompt.Text = [string]$entry.prompt; $prompt.TextWrapping = 'Wrap'; $prompt.Foreground = '#B3BDCB'; $prompt.MaxHeight = 42; $prompt.TextTrimming = 'CharacterEllipsis'
    $meta = New-Object System.Windows.Controls.TextBlock
    $date = [DateTimeOffset]::Parse([string]$entry.created_at).ToLocalTime().ToString('dd/MM/yyyy HH:mm')
    $meta.Text = "$date · $($entry.resolution) · seed $($entry.seed)"; $meta.Foreground = '#7F8B9D'; $meta.FontSize = 12; $meta.Margin = '0,8,0,0'
    $details.Children.Add($model) | Out-Null; $details.Children.Add($prompt) | Out-Null; $details.Children.Add($meta) | Out-Null
    [System.Windows.Controls.Grid]::SetColumn($details, 1); $layout.Children.Add($details) | Out-Null
    $reuse = New-Object System.Windows.Controls.Button
    $reuse.Content = 'Reprendre'; $reuse.Padding = '12,8'; $reuse.VerticalAlignment = 'Center'; $reuse.Tag = $entry
    $reuse.Add_Click({
      param($sender, $eventArgs)
      $item = $sender.Tag
      $script:PromptBox.Text = [string]$item.prompt
      foreach ($choice in $script:ModelBox.Items) { if ($choice.Tag -eq $item.model) { $script:ModelBox.SelectedItem = $choice; break } }
      foreach ($choice in $script:ResolutionBox.Items) { if ($choice.Tag -eq $item.resolution) { $script:ResolutionBox.SelectedItem = $choice; break } }
      Set-Page 'create'
    })
    [System.Windows.Controls.Grid]::SetColumn($reuse, 2); $layout.Children.Add($reuse) | Out-Null
    $card.Child = $layout
    $script:HistoryItems.Children.Add($card) | Out-Null
  }
}

$script:Generator.add_DoWork({
  param($sender, $eventArgs)
  $body = ConvertTo-Json -InputObject $eventArgs.Argument -Depth 8 -Compress
  $eventArgs.Result = Invoke-RestMethod -Uri "$ApiRoot/api/generate" -Method Post -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 3600
})

$script:Generator.add_RunWorkerCompleted({
  param($sender, $eventArgs)
  $script:GenerateButton.IsEnabled = $true
  $script:GenerateButton.Content = "✦   Générer l’image"
  if ($eventArgs.Error) {
    $script:GenerationStatus.Text = $eventArgs.Error.Exception.Message
    $script:GenerationStatus.Foreground = '#E58A8A'
    return
  }
  $item = $eventArgs.Result.item
  try {
    $script:ResultImage.Source = Get-ImageSource $item.image
    $script:ResultImage.Visibility = 'Visible'
    $script:ResultPlaceholder.Visibility = 'Collapsed'
    $script:ResultMeta.Text = "$($item.model_label) · $($item.precision) · $($item.resolution) · seed $($item.seed)"
    $script:GenerationStatus.Text = 'Image générée et enregistrée dans outputs.'
    $script:GenerationStatus.Foreground = '#74D5A6'
  } catch {
    $script:GenerationStatus.Text = "Image enregistrée, mais impossible de l’afficher : $($_.Exception.Message)"
    $script:GenerationStatus.Foreground = '#E5BF77'
  }
})

$script:PromptBox.Add_TextChanged({ param($sender, $eventArgs) $script:CharacterCount.Text = "$($script:PromptBox.Text.Length) / 1000" })
$script:ModelBox.Add_SelectionChanged({
  param($sender, $eventArgs)
  if (-not $script:ModelBox.SelectedItem) { return }
  $isFlux = $script:ModelBox.SelectedItem.Tag -eq 'flux3'
  $script:UpsamplingBox.IsEnabled = $isFlux
  if (-not $isFlux) { $script:UpsamplingBox.IsChecked = $false }
})
$script:NavCreate.Add_Click({ param($sender, $eventArgs) Set-Page 'create' })
$script:NavHistory.Add_Click({ param($sender, $eventArgs) Set-Page 'history' })
$script:NavSettings.Add_Click({ param($sender, $eventArgs) Set-Page 'settings' })
$script:RefreshHistoryButton.Add_Click({ param($sender, $eventArgs) Update-History })
$script:RefreshHealthButton.Add_Click({ param($sender, $eventArgs) Update-Health })

$script:ReferenceButton.Add_Click({
  param($sender, $eventArgs)
  $dialog = New-Object Microsoft.Win32.OpenFileDialog
  $dialog.Title = 'Choisir une image de référence'
  $dialog.Filter = 'Images prises en charge|*.png;*.jpg;*.jpeg;*.webp'
  if ($dialog.ShowDialog($script:window)) {
    $extension = [IO.Path]::GetExtension($dialog.FileName).ToLowerInvariant()
    $mime = switch ($extension) {
      '.png' { 'image/png' }
      '.jpg' { 'image/jpeg' }
      '.jpeg' { 'image/jpeg' }
      '.webp' { 'image/webp' }
      default { $null }
    }
    if (-not $mime) { return }
    $bytes = [IO.File]::ReadAllBytes($dialog.FileName)
    if ($bytes.Length -gt 20MB) {
      [System.Windows.MessageBox]::Show("L’image de référence doit faire 20 Mo ou moins.", 'Image trop volumineuse', [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning) | Out-Null
      return
    }
    $script:ReferenceData = "data:$mime;base64,$([Convert]::ToBase64String($bytes))"
    $script:ReferenceLabel.Text = [IO.Path]::GetFileName($dialog.FileName)
  }
})

$script:GenerateButton.Add_Click({
  param($sender, $eventArgs)
  $prompt = $script:PromptBox.Text.Trim()
  if (-not $prompt) {
    $script:GenerationStatus.Text = 'Écris un prompt avant de lancer la génération.'
    $script:GenerationStatus.Foreground = '#E58A8A'
    return
  }
  $model = [string]$script:ModelBox.SelectedItem.Tag
  $requestBody = @{
    model = $model
    prompt = $prompt
    resolution = [string]$script:ResolutionBox.SelectedItem.Tag
    prompt_upsampling = ($model -eq 'flux3' -and [bool]$script:UpsamplingBox.IsChecked)
    reference_image = $script:ReferenceData
    seed = -1
  }
  $script:GenerateButton.IsEnabled = $false
  $script:GenerateButton.Content = 'Chargement du moteur et génération…'
  $script:GenerationStatus.Text = 'Le chargement du modèle peut prendre plusieurs minutes. Vous pouvez laisser cette fenêtre ouverte.'
  $script:GenerationStatus.Foreground = '#B9C8E8'
  $script:Generator.RunWorkerAsync($requestBody)
})

$script:window = $window
$window.Add_Closed({
  param($sender, $eventArgs)
  if ($script:ServerOwned) {
    try { Invoke-RestMethod -Uri "$ApiRoot/api/shutdown" -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 3 | Out-Null } catch { }
    if ($script:ServerProcess -and -not $script:ServerProcess.HasExited) {
      try { $script:ServerProcess.WaitForExit(4000) | Out-Null } catch { }
    }
  }
})

Update-Health
[void]$window.ShowDialog()
