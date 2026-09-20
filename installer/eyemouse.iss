; Inno Setup script for EyeMouse. Compiled by tools/build_installer.py (passes /DAppVersion=...).
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{B7C6F2A4-3D0E-4C51-9E4B-6F1A2D8E7C10}
AppName=EyeMouse
AppVersion={#AppVersion}
AppVerName=EyeMouse {#AppVersion}
AppPublisher=EyeMouse
DefaultDirName={autopf}\EyeMouse
DefaultGroupName=EyeMouse
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=EyeMouse-Setup-{#AppVersion}
SetupIconFile=..\assets\eyemouse.ico
UninstallDisplayIcon={app}\EyeMouse.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user install by default (no administrator prompt); the dialog lets the user choose "for all users".
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Iniciar o EyeMouse junto com o Windows (o controle do mouse continua desligado até você ligar)"; GroupDescription: "Opções:"; Flags: unchecked

[Files]
Source: "..\build\dist\EyeMouse\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\EyeMouse"; Filename: "{app}\EyeMouse.exe"
Name: "{autodesktop}\EyeMouse"; Filename: "{app}\EyeMouse.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "EyeMouse"; ValueType: string; ValueData: """{app}\EyeMouse.exe"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\EyeMouse.exe"; Description: "{cm:LaunchProgram,EyeMouse}"; Flags: nowait postinstall skipifsilent

[Code]
// Calibration, settings and log live in %APPDATA%\EyeMouse. Keep them on uninstall unless the user says otherwise.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{userappdata}\EyeMouse');
    if (not UninstallSilent()) and DirExists(DataDir) then
      if MsgBox('Remover também a sua calibração, configurações e logs do EyeMouse?' + #13#10 + DataDir,
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
