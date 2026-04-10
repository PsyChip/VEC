{
  VEC Delphi Client SDK

  Usage:
    var Vec: TVecClient;
    Vec := TVecClient.Create('localhost', 1920);
    idx := Vec.Push(myVector);
    results := Vec.Pull(queryVector);
    Vec.Free;

  Works with both TCP (all platforms) and Named Pipes (Windows only).
}
unit vec_client;

interface

uses
  Windows, WinSock, SysUtils;

type
  TVecResult = record
    Index: Integer;
    Distance: Single;
  end;

  TVecResults = array of TVecResult;
  TSingleArray = array of Single;

  TVecClient = class
  private
    FSocket: TSocket;
    FConnected: Boolean;
    FUsePipe: Boolean;
    FPipeHandle: THandle;
    function ReadLine: string;
    procedure SendStr(const S: string);
    procedure SendBytes(const Data: Pointer; Len: Integer);
    function ParseResults(const S: string): TVecResults;
    function VecToCSV(const Vec: TSingleArray): string;
    procedure CheckError(const Resp: string);
  public
    constructor Create; overload;
    function ConnectTCP(const Host: string; Port: Integer = 1920): Boolean;
    function ConnectPipe(const Name: string): Boolean;
    function Push(const Vec: TSingleArray): Integer;
    function Pull(const Vec: TSingleArray): TVecResults;
    function CPull(const Vec: TSingleArray): TVecResults;
    function BPush(const Vectors: array of Single; Count, Dim: Integer): Integer;
    procedure Delete(Index: Integer);
    procedure Undo;
    procedure Save;
    function Size: Integer;
    procedure Close;
    destructor Destroy; override;
    property Connected: Boolean read FConnected;
  end;

implementation

constructor TVecClient.Create;
begin
  inherited;
  FSocket := INVALID_SOCKET;
  FConnected := False;
  FUsePipe := False;
  FPipeHandle := INVALID_HANDLE_VALUE;
end;

function TVecClient.ConnectTCP(const Host: string; Port: Integer): Boolean;
var
  WSA: TWSAData;
  Addr: TSockAddrIn;
  HostEnt: PHostEnt;
begin
  Result := False;
  WSAStartup(MakeWord(2, 2), WSA);

  FSocket := WinSock.socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if FSocket = INVALID_SOCKET then Exit;

  FillChar(Addr, SizeOf(Addr), 0);
  Addr.sin_family := AF_INET;
  Addr.sin_port := htons(Port);
  Addr.sin_addr.S_addr := inet_addr(PAnsiChar(AnsiString(Host)));

  if Addr.sin_addr.S_addr = INADDR_NONE then begin
    HostEnt := gethostbyname(PAnsiChar(AnsiString(Host)));
    if HostEnt = nil then begin
      closesocket(FSocket);
      FSocket := INVALID_SOCKET;
      Exit;
    end;
    Move(HostEnt^.h_addr_list^[0], Addr.sin_addr.S_addr, HostEnt^.h_length);
  end;

  if WinSock.connect(FSocket, Addr, SizeOf(Addr)) <> 0 then begin
    closesocket(FSocket);
    FSocket := INVALID_SOCKET;
    Exit;
  end;

  FConnected := True;
  FUsePipe := False;
  Result := True;
end;

function TVecClient.ConnectPipe(const Name: string): Boolean;
var
  PipeName: string;
begin
  Result := False;
  PipeName := '\\.\pipe\vec_' + Name;

  FPipeHandle := CreateFileA(PAnsiChar(AnsiString(PipeName)),
    GENERIC_READ or GENERIC_WRITE, 0, nil, OPEN_EXISTING, 0, 0);

  if FPipeHandle = INVALID_HANDLE_VALUE then Exit;

  FConnected := True;
  FUsePipe := True;
  Result := True;
end;

procedure TVecClient.SendStr(const S: string);
var
  A: AnsiString;
  Written: DWORD;
begin
  A := AnsiString(S);
  if FUsePipe then
    WriteFile(FPipeHandle, A[1], Length(A), Written, nil)
  else
    WinSock.send(FSocket, A[1], Length(A), 0);
end;

procedure TVecClient.SendBytes(const Data: Pointer; Len: Integer);
var
  Written: DWORD;
  Sent, R: Integer;
begin
  if FUsePipe then begin
    WriteFile(FPipeHandle, Data^, Len, Written, nil);
  end else begin
    Sent := 0;
    while Sent < Len do begin
      R := WinSock.send(FSocket, PAnsiChar(Data)[Sent], Len - Sent, 0);
      if R <= 0 then Break;
      Inc(Sent, R);
    end;
  end;
end;

function TVecClient.ReadLine: string;
var
  C: AnsiChar;
  Buf: AnsiString;
  BytesRead: DWORD;
  R: Integer;
begin
  Buf := '';
  while True do begin
    if FUsePipe then begin
      if not ReadFile(FPipeHandle, C, 1, BytesRead, nil) or (BytesRead = 0) then Break;
    end else begin
      R := recv(FSocket, C, 1, 0);
      if R <= 0 then Break;
    end;
    if C = #10 then Break;
    if C <> #13 then Buf := Buf + C;
  end;
  Result := string(Buf);
end;

function TVecClient.VecToCSV(const Vec: TSingleArray): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to Length(Vec) - 1 do begin
    if I > 0 then Result := Result + ',';
    Result := Result + FormatFloat('0.000000', Vec[I]);
  end;
end;

procedure TVecClient.CheckError(const Resp: string);
begin
  if Copy(Resp, 1, 3) = 'err' then
    raise Exception.Create(Resp);
end;

function TVecClient.ParseResults(const S: string): TVecResults;
var
  Pairs: TArray<string>;
  Parts: TArray<string>;
  I: Integer;
begin
  if (S = '') or (Copy(S, 1, 3) = 'err') then begin
    CheckError(S);
    SetLength(Result, 0);
    Exit;
  end;

  Pairs := S.Split([',']);
  SetLength(Result, Length(Pairs));
  for I := 0 to Length(Pairs) - 1 do begin
    Parts := Pairs[I].Split([':']);
    if Length(Parts) = 2 then begin
      Result[I].Index := StrToIntDef(Parts[0], -1);
      Result[I].Distance := StrToFloatDef(Parts[1], 0);
    end;
  end;
end;

function TVecClient.Push(const Vec: TSingleArray): Integer;
var
  Resp: string;
begin
  SendStr('push ' + VecToCSV(Vec) + #10);
  Resp := ReadLine;
  CheckError(Resp);
  Result := StrToInt(Resp);
end;

function TVecClient.Pull(const Vec: TSingleArray): TVecResults;
begin
  SendStr('pull ' + VecToCSV(Vec) + #10);
  Result := ParseResults(ReadLine);
end;

function TVecClient.CPull(const Vec: TSingleArray): TVecResults;
begin
  SendStr('cpull ' + VecToCSV(Vec) + #10);
  Result := ParseResults(ReadLine);
end;

function TVecClient.BPush(const Vectors: array of Single; Count, Dim: Integer): Integer;
var
  Header, Resp: string;
begin
  Header := 'bpush ' + IntToStr(Count) + #10;
  SendStr(Header);
  SendBytes(@Vectors[0], Count * Dim * SizeOf(Single));
  Resp := ReadLine;
  CheckError(Resp);
  Result := StrToInt(Resp);
end;

procedure TVecClient.Delete(Index: Integer);
begin
  SendStr('delete ' + IntToStr(Index) + #10);
  CheckError(ReadLine);
end;

procedure TVecClient.Undo;
begin
  SendStr('undo' + #10);
  CheckError(ReadLine);
end;

procedure TVecClient.Save;
begin
  SendStr('save' + #10);
  CheckError(ReadLine);
end;

function TVecClient.Size: Integer;
begin
  SendStr('size' + #10);
  Result := StrToInt(ReadLine);
end;

procedure TVecClient.Close;
begin
  if FUsePipe and (FPipeHandle <> INVALID_HANDLE_VALUE) then begin
    FlushFileBuffers(FPipeHandle);
    CloseHandle(FPipeHandle);
    FPipeHandle := INVALID_HANDLE_VALUE;
  end;
  if (not FUsePipe) and (FSocket <> INVALID_SOCKET) then begin
    closesocket(FSocket);
    FSocket := INVALID_SOCKET;
  end;
  FConnected := False;
end;

destructor TVecClient.Destroy;
begin
  Close;
  inherited;
end;

end.
