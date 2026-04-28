{
  VEC 2.0 Delphi Client SDK (binary frame protocol)

  VEC 2.0 is a clean break from 1.x. SDKs from 1.x are not wire-compatible.
  See PROTOCOL-2.0.md for the full spec.

  Usage:
    var Vec: TVecClient;
    var Recs: TVecRecords;
    Vec := TVecClient.Create;
    Vec.ConnectTCP('localhost', 1920);

    // Push: vector required; data requires a label
    idx := Vec.Push(myVector);
    idx := Vec.PushLabeled('img/cat.jpg', myVector);
    idx := Vec.PushFull('img/cat.jpg', myVector, jpegBytes);

    Recs := Vec.Query(qVec);
    Recs := Vec.Query(qVec, True, VEC_SHAPE_LABEL);  // cosine, lean
    Recs := Vec.QID(42);
    Recs := Vec.QIDByLabel('img/cat.jpg', True);

    Recs := Vec.Get(42);
    Recs := Vec.GetByLabel('img/cat.jpg');
    Recs := Vec.GetBatch([0,1,2]);

    Vec.SetData(42, jpegBytes);
    blob := Vec.GetData(42);

    Vec.Update(42, myVector);
    Vec.SetLabel(42, 'img/cat.jpg');
    Vec.Delete(42);                 // also clears label + data
    Vec.Undo;                       // also clears last label + data

    Vec.Save(saved, crc);
    info := Vec.Info;               // info.ProtocolVersion = 2

    Vec.Free;

  Router mode:
    Vec.Namespace := 'tools';

  Works with TCP (all platforms) and Named Pipes (Windows only).
}
unit vec_client;

interface

uses
  Windows, WinSock, SysUtils;

const
  VEC_BIN_MAGIC        = $F0;
  VEC_PROTOCOL_VERSION = $02;

  VEC_CMD_PUSH      = $01;
  VEC_CMD_QUERY     = $02;
  VEC_CMD_GET       = $04;
  VEC_CMD_UPDATE    = $06;
  VEC_CMD_DELETE    = $07;
  VEC_CMD_LABEL     = $08;
  VEC_CMD_UNDO      = $09;
  VEC_CMD_SAVE      = $0A;
  VEC_CMD_CLUSTER   = $0D;
  VEC_CMD_DISTINCT  = $0E;
  VEC_CMD_REPRESENT = $0F;
  VEC_CMD_INFO      = $10;
  VEC_CMD_QID       = $11;
  VEC_CMD_SET_DATA  = $13;
  VEC_CMD_GET_DATA  = $14;

  VEC_RESP_OK  = $00;
  VEC_RESP_ERR = $01;

  VEC_SHAPE_VECTOR = $01;
  VEC_SHAPE_LABEL  = $02;
  VEC_SHAPE_DATA   = $04;
  VEC_SHAPE_FULL   = VEC_SHAPE_VECTOR or VEC_SHAPE_LABEL or VEC_SHAPE_DATA;

  VEC_GET_MODE_SINGLE = $00;
  VEC_GET_MODE_BATCH  = $01;

  VEC_METRIC_L2     = $00;
  VEC_METRIC_COSINE = $01;

  VEC_MAX_LABEL_BYTES = 2048;
  VEC_MAX_DATA_BYTES  = 102400;

type
  TSingleArray  = array of Single;
  TByteArray    = array of Byte;

  TVecRecord = record
    Index    : Integer;
    Distance : Single;       { valid only for query/qid }
    Label_   : string;       { empty if shape excluded label }
    Data     : TByteArray;   { nil if shape excluded data }
    Vector   : TSingleArray; { nil if shape excluded vector }
  end;

  TVecRecords = array of TVecRecord;

  TVecInfo = record
    Dim             : Integer;
    Count           : Integer;
    Deleted         : Integer;
    Fmt             : Integer;     { 0=f32, 1=f16 }
    MTime           : Int64;
    CRC             : Cardinal;
    CRC_OK          : Integer;     { 0=mismatch, 1=ok, 2=unknown }
    Name            : string;
    ProtocolVersion : Integer;
  end;

  TVecClient = class
  private
    FSocket: TSocket;
    FConnected: Boolean;
    FUsePipe: Boolean;
    FPipeHandle: THandle;
    FNamespace: string;
    FDimCache: Integer;
    procedure SendRaw(const Data: Pointer; Len: Integer);
    procedure RecvExact(Buf: Pointer; Len: Integer);
    procedure SendFrame(Cmd: Byte; const ALabel: AnsiString; const Body: AnsiString);
    function RecvResponse: AnsiString;
    function ParseRecords(const Body: AnsiString; Shape: Byte; Dim: Integer;
                          WithDistance: Boolean): TVecRecords;
    function EnsureDim: Integer;
  public
    constructor Create; overload;
    function ConnectTCP(const Host: string; Port: Integer = 1920): Boolean;
    function ConnectPipe(const Name: string): Boolean;

    function Push(const Vec: TSingleArray): Integer;
    function PushLabeled(const ALabel: string; const Vec: TSingleArray): Integer;
    function PushFull(const ALabel: string; const Vec: TSingleArray; const Data: TByteArray): Integer;

    function Query(const Vec: TSingleArray): TVecRecords; overload;
    function Query(const Vec: TSingleArray; Cosine: Boolean; Shape: Byte = VEC_SHAPE_FULL): TVecRecords; overload;

    function QID(Index: Integer): TVecRecords; overload;
    function QID(Index: Integer; Cosine: Boolean; Shape: Byte = VEC_SHAPE_FULL): TVecRecords; overload;
    function QIDByLabel(const ALabel: string): TVecRecords; overload;
    function QIDByLabel(const ALabel: string; Cosine: Boolean; Shape: Byte = VEC_SHAPE_FULL): TVecRecords; overload;

    function Get(Index: Integer; Shape: Byte = VEC_SHAPE_FULL): TVecRecords;
    function GetByLabel(const ALabel: string; Shape: Byte = VEC_SHAPE_FULL): TVecRecords;
    function GetBatch(const Indices: array of Integer; Shape: Byte = VEC_SHAPE_FULL): TVecRecords;

    procedure SetData(Index: Integer; const Data: TByteArray); overload;
    procedure SetData(const ALabel: string; const Data: TByteArray); overload;
    function GetData(Index: Integer): TByteArray; overload;
    function GetData(const ALabel: string): TByteArray; overload;

    procedure Update(Index: Integer; const Vec: TSingleArray);
    procedure UpdateByLabel(const ALabel: string; const Vec: TSingleArray);
    procedure SetLabel(Index: Integer; const ALabel: string);
    procedure Delete(Index: Integer); overload;
    procedure Delete(const ALabel: string); overload;
    procedure Undo;
    procedure Save(out SavedCount: Cardinal; out CRC: Cardinal); overload;
    procedure Save; overload;
    function Info: TVecInfo;

    function ClusterRaw(Eps: Single; Cosine: Boolean = False; MinPts: Integer = 2): TArray<string>;
    function DistinctRaw(K: Integer; Cosine: Boolean = False): TArray<string>;
    function RepresentRaw(Eps: Single; Cosine: Boolean = False; MinPts: Integer = 2): TArray<string>;

    procedure Close;
    destructor Destroy; override;
    property Connected: Boolean read FConnected;
    property Namespace: string read FNamespace write FNamespace;
  end;

implementation

constructor TVecClient.Create;
begin
  inherited;
  FSocket := INVALID_SOCKET;
  FConnected := False;
  FUsePipe := False;
  FPipeHandle := INVALID_HANDLE_VALUE;
  FNamespace := '';
  FDimCache := 0;
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

procedure TVecClient.SendRaw(const Data: Pointer; Len: Integer);
var
  Written: DWORD;
  Sent, R: Integer;
begin
  if Len <= 0 then Exit;
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

procedure TVecClient.RecvExact(Buf: Pointer; Len: Integer);
var
  BytesRead: DWORD;
  Got, R: Integer;
begin
  if Len <= 0 then Exit;
  if FUsePipe then begin
    Got := 0;
    while Got < Len do begin
      ReadFile(FPipeHandle, PAnsiChar(Buf)[Got], Len - Got, BytesRead, nil);
      if BytesRead = 0 then Break;
      Inc(Got, BytesRead);
    end;
  end else begin
    Got := 0;
    while Got < Len do begin
      R := recv(FSocket, PAnsiChar(Buf)[Got], Len - Got, 0);
      if R <= 0 then Break;
      Inc(Got, R);
    end;
  end;
end;

procedure TVecClient.SendFrame(Cmd: Byte; const ALabel: AnsiString; const Body: AnsiString);
var
  NS: AnsiString;
  Frame: AnsiString;
  NSLen, LblLen: Word;
  BodyLen: Cardinal;
  Off: Integer;
begin
  if FNamespace <> '' then NS := AnsiString(FNamespace) else NS := '';
  NSLen := Length(NS);
  LblLen := Length(ALabel);
  if LblLen > VEC_MAX_LABEL_BYTES then
    raise Exception.CreateFmt('label too long (%d > %d)', [LblLen, VEC_MAX_LABEL_BYTES]);
  BodyLen := Length(Body);

  SetLength(Frame, 1 + 2 + NSLen + 1 + 2 + LblLen + 4 + BodyLen);
  Off := 1;
  Frame[Off] := AnsiChar(VEC_BIN_MAGIC); Inc(Off);
  Move(NSLen, Frame[Off], 2); Inc(Off, 2);
  if NSLen > 0 then begin Move(NS[1], Frame[Off], NSLen); Inc(Off, NSLen); end;
  Frame[Off] := AnsiChar(Cmd); Inc(Off);
  Move(LblLen, Frame[Off], 2); Inc(Off, 2);
  if LblLen > 0 then begin Move(ALabel[1], Frame[Off], LblLen); Inc(Off, LblLen); end;
  Move(BodyLen, Frame[Off], 4); Inc(Off, 4);
  if BodyLen > 0 then Move(Body[1], Frame[Off], BodyLen);

  SendRaw(@Frame[1], Length(Frame));
end;

function TVecClient.RecvResponse: AnsiString;
var
  Hdr: array[0..4] of Byte;
  Status: Byte;
  BodyLen: Cardinal;
  Body: AnsiString;
begin
  RecvExact(@Hdr[0], 5);
  Status := Hdr[0];
  Move(Hdr[1], BodyLen, 4);
  Body := '';
  if BodyLen > 0 then begin
    SetLength(Body, BodyLen);
    RecvExact(@Body[1], BodyLen);
  end;
  if Status = VEC_RESP_ERR then
    raise Exception.Create(string(Body));
  if Status <> VEC_RESP_OK then
    raise Exception.CreateFmt('unknown status byte 0x%02x', [Status]);
  Result := Body;
end;

function TVecClient.ParseRecords(const Body: AnsiString; Shape: Byte; Dim: Integer;
                                 WithDistance: Boolean): TVecRecords;
var
  Count: Cardinal;
  Off: Integer;
  I: Integer;
  Idx: Integer;
  Dist: Single;
  LL, DL: Cardinal;
  LbBuf: AnsiString;
  DBuf: TByteArray;
  V: TSingleArray;
begin
  if Length(Body) < 4 then begin SetLength(Result, 0); Exit; end;
  Move(Body[1], Count, 4);
  SetLength(Result, Count);
  Off := 5; { 1-based + 4 byte count prefix }
  for I := 0 to Integer(Count) - 1 do begin
    Move(Body[Off], Idx, 4); Inc(Off, 4);
    Result[I].Index := Idx;
    Dist := 0;
    if WithDistance then begin
      Move(Body[Off], Dist, 4); Inc(Off, 4);
    end;
    Result[I].Distance := Dist;
    Result[I].Label_ := '';
    SetLength(Result[I].Data, 0);
    SetLength(Result[I].Vector, 0);

    if (Shape and VEC_SHAPE_LABEL) <> 0 then begin
      Move(Body[Off], LL, 4); Inc(Off, 4);
      if LL > 0 then begin
        SetLength(LbBuf, LL);
        Move(Body[Off], LbBuf[1], LL);
        Inc(Off, LL);
        Result[I].Label_ := string(LbBuf);
      end;
    end;
    if (Shape and VEC_SHAPE_DATA) <> 0 then begin
      Move(Body[Off], DL, 4); Inc(Off, 4);
      if DL > 0 then begin
        SetLength(DBuf, DL);
        Move(Body[Off], DBuf[0], DL);
        Inc(Off, Integer(DL));
        Result[I].Data := DBuf;
      end;
    end;
    if (Shape and VEC_SHAPE_VECTOR) <> 0 then begin
      SetLength(V, Dim);
      Move(Body[Off], V[0], Dim * SizeOf(Single));
      Inc(Off, Dim * SizeOf(Single));
      Result[I].Vector := V;
    end;
  end;
end;

function TVecClient.EnsureDim: Integer;
var I: TVecInfo;
begin
  if FDimCache > 0 then Exit(FDimCache);
  I := Info;
  FDimCache := I.Dim;
  Result := FDimCache;
end;

{ ---------------- PUSH ---------------- }

function TVecClient.Push(const Vec: TSingleArray): Integer;
var Body: AnsiString; Zero: Cardinal; VBytes: Integer;
begin
  VBytes := Length(Vec) * SizeOf(Single);
  SetLength(Body, VBytes + 4);
  if VBytes > 0 then Move(Vec[0], Body[1], VBytes);
  Zero := 0;
  Move(Zero, Body[VBytes + 1], 4);
  SendFrame(VEC_CMD_PUSH, '', Body);
  Body := RecvResponse;
  Move(Body[1], Result, 4);
end;

function TVecClient.PushLabeled(const ALabel: string; const Vec: TSingleArray): Integer;
var Body: AnsiString; Zero: Cardinal; VBytes: Integer;
begin
  VBytes := Length(Vec) * SizeOf(Single);
  SetLength(Body, VBytes + 4);
  if VBytes > 0 then Move(Vec[0], Body[1], VBytes);
  Zero := 0;
  Move(Zero, Body[VBytes + 1], 4);
  SendFrame(VEC_CMD_PUSH, AnsiString(ALabel), Body);
  Body := RecvResponse;
  Move(Body[1], Result, 4);
end;

function TVecClient.PushFull(const ALabel: string; const Vec: TSingleArray; const Data: TByteArray): Integer;
var Body, R: AnsiString; DLen: Cardinal; VBytes: Integer;
begin
  if (Length(Data) > 0) and (ALabel = '') then
    raise Exception.Create('data requires label');
  if Length(Data) > VEC_MAX_DATA_BYTES then
    raise Exception.CreateFmt('data too large (%d > %d)', [Length(Data), VEC_MAX_DATA_BYTES]);
  VBytes := Length(Vec) * SizeOf(Single);
  DLen := Length(Data);
  SetLength(Body, VBytes + 4 + Integer(DLen));
  if VBytes > 0 then Move(Vec[0], Body[1], VBytes);
  Move(DLen, Body[VBytes + 1], 4);
  if DLen > 0 then Move(Data[0], Body[VBytes + 5], DLen);
  SendFrame(VEC_CMD_PUSH, AnsiString(ALabel), Body);
  R := RecvResponse;
  Move(R[1], Result, 4);
end;

{ ---------------- QUERY ---------------- }

function TVecClient.Query(const Vec: TSingleArray): TVecRecords;
begin
  Result := Query(Vec, False, VEC_SHAPE_FULL);
end;

function TVecClient.Query(const Vec: TSingleArray; Cosine: Boolean; Shape: Byte): TVecRecords;
var Body, R: AnsiString; VBytes: Integer; Dim: Integer;
begin
  Dim := EnsureDim;
  VBytes := Length(Vec) * SizeOf(Single);
  SetLength(Body, 2 + VBytes);
  if Cosine then Body[1] := AnsiChar(VEC_METRIC_COSINE) else Body[1] := AnsiChar(VEC_METRIC_L2);
  Body[2] := AnsiChar(Shape);
  if VBytes > 0 then Move(Vec[0], Body[3], VBytes);
  SendFrame(VEC_CMD_QUERY, '', Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, True);
end;

{ ---------------- QID ---------------- }

function TVecClient.QID(Index: Integer): TVecRecords;
begin
  Result := QID(Index, False, VEC_SHAPE_FULL);
end;

function TVecClient.QID(Index: Integer; Cosine: Boolean; Shape: Byte): TVecRecords;
var Body, R: AnsiString; Dim: Integer;
begin
  Dim := EnsureDim;
  SetLength(Body, 6);
  if Cosine then Body[1] := AnsiChar(VEC_METRIC_COSINE) else Body[1] := AnsiChar(VEC_METRIC_L2);
  Body[2] := AnsiChar(Shape);
  Move(Index, Body[3], 4);
  SendFrame(VEC_CMD_QID, '', Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, True);
end;

function TVecClient.QIDByLabel(const ALabel: string): TVecRecords;
begin
  Result := QIDByLabel(ALabel, False, VEC_SHAPE_FULL);
end;

function TVecClient.QIDByLabel(const ALabel: string; Cosine: Boolean; Shape: Byte): TVecRecords;
var Body, R: AnsiString; Dim: Integer;
begin
  Dim := EnsureDim;
  SetLength(Body, 2);
  if Cosine then Body[1] := AnsiChar(VEC_METRIC_COSINE) else Body[1] := AnsiChar(VEC_METRIC_L2);
  Body[2] := AnsiChar(Shape);
  SendFrame(VEC_CMD_QID, AnsiString(ALabel), Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, True);
end;

{ ---------------- GET ---------------- }

function TVecClient.Get(Index: Integer; Shape: Byte): TVecRecords;
var Body, R: AnsiString; Dim: Integer;
begin
  Dim := EnsureDim;
  SetLength(Body, 6);
  Body[1] := AnsiChar(VEC_GET_MODE_SINGLE);
  Body[2] := AnsiChar(Shape);
  Move(Index, Body[3], 4);
  SendFrame(VEC_CMD_GET, '', Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, False);
end;

function TVecClient.GetByLabel(const ALabel: string; Shape: Byte): TVecRecords;
var Body, R: AnsiString; Dim: Integer;
begin
  Dim := EnsureDim;
  SetLength(Body, 2);
  Body[1] := AnsiChar(VEC_GET_MODE_SINGLE);
  Body[2] := AnsiChar(Shape);
  SendFrame(VEC_CMD_GET, AnsiString(ALabel), Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, False);
end;

function TVecClient.GetBatch(const Indices: array of Integer; Shape: Byte): TVecRecords;
var Body, R: AnsiString; Dim, N, I: Integer; Cnt: Cardinal;
begin
  Dim := EnsureDim;
  N := Length(Indices);
  Cnt := N;
  SetLength(Body, 2 + 4 + N * 4);
  Body[1] := AnsiChar(VEC_GET_MODE_BATCH);
  Body[2] := AnsiChar(Shape);
  Move(Cnt, Body[3], 4);
  for I := 0 to N - 1 do
    Move(Indices[I], Body[7 + I * 4], 4);
  SendFrame(VEC_CMD_GET, '', Body);
  R := RecvResponse;
  Result := ParseRecords(R, Shape, Dim, False);
end;

{ ---------------- SET_DATA / GET_DATA ---------------- }

procedure TVecClient.SetData(Index: Integer; const Data: TByteArray);
var Body: AnsiString; DLen: Cardinal;
begin
  if Length(Data) > VEC_MAX_DATA_BYTES then
    raise Exception.CreateFmt('data too large (%d > %d)', [Length(Data), VEC_MAX_DATA_BYTES]);
  DLen := Length(Data);
  SetLength(Body, 8 + Integer(DLen));
  Move(Index, Body[1], 4);
  Move(DLen, Body[5], 4);
  if DLen > 0 then Move(Data[0], Body[9], DLen);
  SendFrame(VEC_CMD_SET_DATA, '', Body);
  RecvResponse;
end;

procedure TVecClient.SetData(const ALabel: string; const Data: TByteArray);
var Body: AnsiString; DLen: Cardinal;
begin
  if Length(Data) > VEC_MAX_DATA_BYTES then
    raise Exception.CreateFmt('data too large (%d > %d)', [Length(Data), VEC_MAX_DATA_BYTES]);
  DLen := Length(Data);
  SetLength(Body, 4 + Integer(DLen));
  Move(DLen, Body[1], 4);
  if DLen > 0 then Move(Data[0], Body[5], DLen);
  SendFrame(VEC_CMD_SET_DATA, AnsiString(ALabel), Body);
  RecvResponse;
end;

function TVecClient.GetData(Index: Integer): TByteArray;
var Body, R: AnsiString; DLen: Cardinal;
begin
  SetLength(Body, 4);
  Move(Index, Body[1], 4);
  SendFrame(VEC_CMD_GET_DATA, '', Body);
  R := RecvResponse;
  Move(R[1], DLen, 4);
  SetLength(Result, DLen);
  if DLen > 0 then Move(R[5], Result[0], DLen);
end;

function TVecClient.GetData(const ALabel: string): TByteArray;
var R: AnsiString; DLen: Cardinal;
begin
  SendFrame(VEC_CMD_GET_DATA, AnsiString(ALabel), '');
  R := RecvResponse;
  Move(R[1], DLen, 4);
  SetLength(Result, DLen);
  if DLen > 0 then Move(R[5], Result[0], DLen);
end;

{ ---------------- UPDATE / LABEL / DELETE / UNDO ---------------- }

procedure TVecClient.Update(Index: Integer; const Vec: TSingleArray);
var Body: AnsiString; VBytes: Integer;
begin
  VBytes := Length(Vec) * SizeOf(Single);
  SetLength(Body, 4 + VBytes);
  Move(Index, Body[1], 4);
  if VBytes > 0 then Move(Vec[0], Body[5], VBytes);
  SendFrame(VEC_CMD_UPDATE, '', Body);
  RecvResponse;
end;

procedure TVecClient.UpdateByLabel(const ALabel: string; const Vec: TSingleArray);
var Body: AnsiString; VBytes: Integer;
begin
  VBytes := Length(Vec) * SizeOf(Single);
  SetLength(Body, VBytes);
  if VBytes > 0 then Move(Vec[0], Body[1], VBytes);
  SendFrame(VEC_CMD_UPDATE, AnsiString(ALabel), Body);
  RecvResponse;
end;

procedure TVecClient.SetLabel(Index: Integer; const ALabel: string);
var Body: AnsiString;
begin
  SetLength(Body, 4);
  Move(Index, Body[1], 4);
  SendFrame(VEC_CMD_LABEL, AnsiString(ALabel), Body);
  RecvResponse;
end;

procedure TVecClient.Delete(Index: Integer);
var Body: AnsiString;
begin
  SetLength(Body, 4);
  Move(Index, Body[1], 4);
  SendFrame(VEC_CMD_DELETE, '', Body);
  RecvResponse;
end;

procedure TVecClient.Delete(const ALabel: string);
begin
  SendFrame(VEC_CMD_DELETE, AnsiString(ALabel), '');
  RecvResponse;
end;

procedure TVecClient.Undo;
begin
  SendFrame(VEC_CMD_UNDO, '', '');
  RecvResponse;
end;

procedure TVecClient.Save(out SavedCount: Cardinal; out CRC: Cardinal);
var R: AnsiString;
begin
  SendFrame(VEC_CMD_SAVE, '', '');
  R := RecvResponse;
  Move(R[1], SavedCount, 4);
  Move(R[5], CRC, 4);
end;

procedure TVecClient.Save;
var Saved, CRC: Cardinal;
begin
  Save(Saved, CRC);
end;

function TVecClient.Info: TVecInfo;
var R: AnsiString; Off: Integer; NameLen: Cardinal; NameBuf: AnsiString;
begin
  SendFrame(VEC_CMD_INFO, '', '');
  R := RecvResponse;
  Off := 1;
  Move(R[Off], Result.Dim, 4);     Inc(Off, 4);
  Move(R[Off], Result.Count, 4);   Inc(Off, 4);
  Move(R[Off], Result.Deleted, 4); Inc(Off, 4);
  Result.Fmt := Byte(R[Off]);      Inc(Off);
  Move(R[Off], Result.MTime, 8);   Inc(Off, 8);
  Move(R[Off], Result.CRC, 4);     Inc(Off, 4);
  Result.CRC_OK := Byte(R[Off]);   Inc(Off);
  Move(R[Off], NameLen, 4);        Inc(Off, 4);
  if NameLen > 0 then begin
    SetLength(NameBuf, NameLen);
    Move(R[Off], NameBuf[1], NameLen);
    Result.Name := string(NameBuf);
    Inc(Off, NameLen);
  end else
    Result.Name := '';
  Result.ProtocolVersion := Byte(R[Off]);
end;

{ ---------------- CLUSTER / DISTINCT / REPRESENT (legacy text bodies) ---------------- }

function TVecClient.ClusterRaw(Eps: Single; Cosine: Boolean; MinPts: Integer): TArray<string>;
var Body, R: AnsiString; Mode: Byte; Lines: TArray<string>; SS: TArray<string>; I: Integer;
begin
  SetLength(Body, 9);
  Move(Eps, Body[1], 4);
  if Cosine then Mode := VEC_METRIC_COSINE else Mode := VEC_METRIC_L2;
  Body[5] := AnsiChar(Mode);
  Move(MinPts, Body[6], 4);
  SendFrame(VEC_CMD_CLUSTER, '', Body);
  R := RecvResponse;
  SS := string(R).Split([#10]);
  SetLength(Lines, 0);
  for I := 0 to Length(SS) - 1 do begin
    if (SS[I] = '') or (SS[I] = 'end') then Continue;
    SetLength(Lines, Length(Lines) + 1);
    Lines[High(Lines)] := SS[I];
  end;
  Result := Lines;
end;

function TVecClient.DistinctRaw(K: Integer; Cosine: Boolean): TArray<string>;
var Body, R: AnsiString; Mode: Byte; Lines: TArray<string>; SS: TArray<string>; I: Integer;
begin
  SetLength(Body, 5);
  Move(K, Body[1], 4);
  if Cosine then Mode := VEC_METRIC_COSINE else Mode := VEC_METRIC_L2;
  Body[5] := AnsiChar(Mode);
  SendFrame(VEC_CMD_DISTINCT, '', Body);
  R := RecvResponse;
  SS := string(R).Split([#10]);
  SetLength(Lines, 0);
  for I := 0 to Length(SS) - 1 do begin
    if (SS[I] = '') or (SS[I] = 'end') then Continue;
    SetLength(Lines, Length(Lines) + 1);
    Lines[High(Lines)] := SS[I];
  end;
  Result := Lines;
end;

function TVecClient.RepresentRaw(Eps: Single; Cosine: Boolean; MinPts: Integer): TArray<string>;
var Body, R: AnsiString; Mode: Byte; Lines: TArray<string>; SS: TArray<string>; I: Integer;
begin
  SetLength(Body, 9);
  Move(Eps, Body[1], 4);
  if Cosine then Mode := VEC_METRIC_COSINE else Mode := VEC_METRIC_L2;
  Body[5] := AnsiChar(Mode);
  Move(MinPts, Body[6], 4);
  SendFrame(VEC_CMD_REPRESENT, '', Body);
  R := RecvResponse;
  SS := string(R).Split([#10]);
  SetLength(Lines, 0);
  for I := 0 to Length(SS) - 1 do begin
    if (SS[I] = '') or (SS[I] = 'end') then Continue;
    SetLength(Lines, Length(Lines) + 1);
    Lines[High(Lines)] := SS[I];
  end;
  Result := Lines;
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
