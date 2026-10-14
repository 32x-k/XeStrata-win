// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// xestrata.exe: the window, after the check that the 7z is complete and the VC++ runtime is installed (said in a
/// message box of Windows' own, as the window's library may need what is missing).
module Xestrata.Launcher.Program

open System
open System.Runtime.InteropServices
open Avalonia
open Avalonia.Controls.ApplicationLifetimes
open Avalonia.Themes.Fluent

[<DllImport("user32.dll", CharSet = CharSet.Unicode)>]
extern int MessageBoxW(nativeint hWnd, string text, string caption, uint32 kind)

type App() =
    inherit Application()

    override this.Initialize() = this.Styles.Add(FluentTheme())

    override this.OnFrameworkInitializationCompleted() =
        match this.ApplicationLifetime with
        | :? IClassicDesktopStyleApplicationLifetime as desktop -> desktop.MainWindow <- new MainWindow.MainWindow()
        | _ -> ()
        base.OnFrameworkInitializationCompleted()

[<EntryPoint; STAThread>]
let main argv =
    match Runtime.problem () with
    | Some why ->
        MessageBoxW(0n, why, "XeStrata", 0x10u) |> ignore   // MB_ICONERROR
        1
    | None -> AppBuilder.Configure<App>().UsePlatformDetect().StartWithClassicDesktopLifetime argv
