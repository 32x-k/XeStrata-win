// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The launcher's window: this PC and the installed models (setup.py --describe), the choice of a model to install,
/// the install's progress and questions, and the running model.  setup.py does the work; the window shows it.
module Xestrata.Launcher.MainWindow

open System
open System.Diagnostics
open System.Text.RegularExpressions
open Avalonia
open Avalonia.Controls
open Avalonia.Controls.Templates
open Avalonia.Layout
open Avalonia.Platform.Storage
open Avalonia.Threading
open Xestrata.Launcher.Protocol
open Xestrata.Launcher.Views

let private ui (f: unit -> unit) = Dispatcher.UIThread.Post(fun () -> f ())

let private openUrl (url: string) =
    Process.Start(ProcessStartInfo(url, UseShellExecute = true)) |> ignore

let private readyUrl = Regex(@"ready: (http://\S+?)(/v1)?\s", RegexOptions.Compiled)

/// The log of a run, its last lines kept.
type private LogView() =
    let box =
        TextBox(
            IsReadOnly = true,
            AcceptsReturn = true,
            TextWrapping = Media.TextWrapping.NoWrap,
            FontFamily = Media.FontFamily "Cascadia Mono, Consolas, monospace",
            FontSize = 12.0,
            Height = 220.0
        )
    let lines = Collections.Generic.Queue<string>()

    member _.Control = box

    member _.Add(line: string) =
        lines.Enqueue line
        while lines.Count > 400 do
            lines.Dequeue() |> ignore
        box.Text <- String.Join("\n", lines)
        box.CaretIndex <- box.Text.Length

type MainWindow() as this =
    inherit Window(Title = "XeStrata", Width = 980.0, Height = 720.0, MinWidth = 720.0, MinHeight = 520.0)

    let host = ContentControl(Margin = Thickness(24.0, 16.0))
    let version = muted ""
    let updateBar = Border(IsVisible = false, Background = accentTint, CornerRadius = CornerRadius 6.0, Padding = Thickness 10.0)
    let mutable running: Setup.Run option = None

    let show (content: Control) =
        host.Content <- ScrollViewer(Content = content, HorizontalScrollBarVisibility = Primitives.ScrollBarVisibility.Disabled)

    let failure (title: string) (detail: string) (back: unit -> unit) =
        show (
            stack 14.0 [
                heading title
                card (coloured failBrush detail)
                actions [ accentButton "Back" back ]
            ]
        )

    let pickFolder (title: string) (target: TextBox) =
        task {
            let top = TopLevel.GetTopLevel this
            let! picked = top.StorageProvider.OpenFolderPickerAsync(FolderPickerOpenOptions(Title = title, AllowMultiple = false))
            if picked.Count > 0 then
                match picked[0].TryGetLocalPath() with
                | null -> ()
                | path -> target.Text <- path
        }
        |> ignore

    // ---------------------------------------------------------------------------------------------- the update
    let checkUpdate () =
        task {
            let! messages, _ = Setup.collect [ "--check-update" ]
            for m in messages do
                match m with
                | Updated u when u.Newer ->
                    ui (fun () ->
                        updateBar.Child <-
                            stack 6.0 [
                                subheading (sprintf "XeStrata %s is out (this is %s)" u.Latest u.Current)
                                text ("To update: " + u.How + ".")
                                actions [ button "Release page" (fun () -> openUrl u.Url) ]
                            ]
                        updateBar.IsVisible <- true)
                | _ -> ()
        }
        |> ignore

    // ---------------------------------------------------------------------------------------------- a run of setup
    /// setup.py with `args`, its steps, progress, questions and log on a page of its own: `answer` answers a question
    /// the window has already asked (else the page asks it), `onReady` gets the server's address once it says so,
    /// `onDone` the exit code (after a failure: when the page is left).
    let runPage (title: string) (args: string list) (answer: string -> string option) (onReady: string -> unit)
                (onDone: int -> unit) =
        let stepText = subheading "Starting ..."
        let status = text ""
        let bar = ProgressBar(IsIndeterminate = true, Minimum = 0.0, Maximum = 1.0, Height = 6.0)
        let warnings = stack 4.0 []
        let question = stack 8.0 []
        question.IsVisible <- false
        let log = LogView()
        let recent = Collections.Generic.List<string>()   // the says since the last question: a question's menu
        let details = Expander(Header = "Details", Content = log.Control, HorizontalAlignment = HorizontalAlignment.Stretch)
        let mutable run: Setup.Run option = None
        let mutable failed = false
        let cancel = button "Cancel" (fun () -> run |> Option.iter (fun r -> r.Stop()))
        let mutable readySeen = false
        let line (s: string) =
            log.Add s
            if not readySeen then
                let m = readyUrl.Match(s + " ")
                if m.Success then
                    readySeen <- true
                    onReady m.Groups[1].Value
        let onMessage (m: Message) =
            ui (fun () ->
                match m with
                | Step(n, t) ->
                    stepText.Text <- sprintf "Step %d: %s" n t
                    bar.IsIndeterminate <- true
                    recent.Clear()
                    log.Add("=== " + t)
                | Say s ->
                    status.Text <- s.Trim()
                    if s.Trim() <> "" then recent.Add s
                    line s
                | Ok s ->
                    status.Text <- "✓ " + s
                    line ("[ok] " + s)
                | Warn s ->
                    warnings.Children.Add(coloured warnBrush ("⚠ " + s))
                    line ("[!] " + s)
                | Fail(s, hint) ->
                    failed <- true
                    failure "Setup stopped" (if String.IsNullOrEmpty hint then s else s + "\n\n" + hint) (fun () -> onDone 1)
                | Progress(what, d, total) ->
                    if total > 0L then
                        bar.IsIndeterminate <- false
                        bar.Value <- float d / float total
                        status.Text <- sprintf "%s: %s of %s" what (gb (float d / 1e9)) (gb (float total / 1e9))
                    else
                        status.Text <- sprintf "%s: %s" what (gb (float d / 1e9))
                | Log s
                | Text s -> line s
                | Ask(q, _, _) when (answer q).IsSome -> run |> Option.iter (fun r -> r.Answer (answer q).Value)
                | Ask(q, choices, def) ->
                    question.Children.Clear()
                    question.Children.Add(subheading q)
                    for r in recent do
                        question.Children.Add(text (r.TrimEnd()))
                    let buttons =
                        [ for c in choices ->
                              let answer () =
                                  question.IsVisible <- false
                                  recent.Clear()
                                  run |> Option.iter (fun r -> r.Answer c)
                              (if c = def then accentButton c answer else button c answer) :> Control ]
                    question.Children.Add(row 8.0 buttons)
                    question.IsVisible <- true
                | Described _
                | Updated _ -> ())
        show (
            stack 14.0 [
                heading title
                stepText
                bar
                status
                warnings
                card question
                details
                actions [ cancel ]
            ]
        )
        let r = Setup.Run(args, onMessage, (fun code -> ui (fun () -> if not failed then onDone code)))
        run <- Some r
        r

    // ---------------------------------------------------------------------------------------------- the pages
    let rec home () =
        show (stack 10.0 [ heading "Checking this PC ..."; ProgressBar(IsIndeterminate = true) ])
        task {
            let! messages, code = Setup.collect [ "--describe" ]
            let found = messages |> List.tryPick (function Described d -> Some d | _ -> None)
            ui (fun () ->
                match found with
                | Some d -> homePage d
                | None ->
                    let why =
                        messages
                        |> List.choose (function Fail(t, h) -> Some(t + " " + h) | Text t | Log t -> Some t | _ -> None)
                        |> String.concat "\n"
                    failure "XeStrata could not check this PC" (sprintf "setup.py exited with %d.\n%s" code why) home)
        }
        |> ignore

    and homePage (d: Describe) =
        version.Text <- "xe" + d.Version
        let gpu =
            if d.Gpu.HasValue then
                d.Gpus |> Array.tryFind (fun g -> g.Index = d.Gpu.Value)
            else
                None
        let pc =
            stack 6.0 [
                yield subheading "This PC"
                match gpu with
                | Some g -> yield text (sprintf "GPU: %s, %s VRAM" g.Name (gb g.VramGb))
                | None -> yield coloured failBrush "No GPU here can run XeStrata."
                for g in d.Gpus do
                    if not (isNull g.Problem) then
                        yield coloured warnBrush (sprintf "%s: %s" g.Name g.Problem)
                yield text (sprintf "RAM: %s" (gb d.RamGb))
                yield
                    text (
                        sprintf "CPU: %s%s" d.Cpu.Name (if d.Cpu.Avx512 then " (AVX-512)" elif d.Cpu.Avx2 then " (AVX2)" else " (no AVX2: not supported)")
                    )
            ]
        let installed =
            [ for i, m in Array.indexed d.Installed ->
                  card (
                      let g = Grid(ColumnDefinitions = ColumnDefinitions "*,Auto")
                      let info =
                          stack 2.0 [
                              subheading m.ModelName
                              muted (
                                  (if m.Context.HasValue then contextLabel m.Context.Value + " of context, " else "")
                                  + sprintf "http://127.0.0.1:%d/" m.Port
                              )
                          ]
                      let start = accentButton "Start" (fun () -> startModel i m)
                      Grid.SetColumn(start, 1)
                      start.VerticalAlignment <- VerticalAlignment.Center
                      g.Children.Add info
                      g.Children.Add start
                      g
                  )
                  :> Control ]
        show (
            stack 14.0 [
                yield card pc :> Control
                if not installed.IsEmpty then
                    yield subheading "Installed models"
                    yield! installed
                yield
                    actions [
                        (if installed.IsEmpty then accentButton else button) "Install a model ..." (fun () -> choosePage d)
                    ]
            ]
        )

    and choosePage (d: Describe) =
        let families = ListBox(ItemsSource = d.Families, MaxHeight = 260.0)
        families.ItemTemplate <-
            FuncDataTemplate<Family>(fun f _ ->
                stack 2.0 [
                    row 6.0 [
                        subheading f.Title
                        if f.Experimental then badge warnBrush "experimental"
                    ]
                    muted (f.By + " - " + f.About)
                ])
        let sizes = ListBox(MaxHeight = 320.0)
        sizes.ItemTemplate <-
            FuncDataTemplate<Model>(fun m _ ->
                stack 2.0 [
                    row 6.0 [
                        yield subheading m.Id
                        if m.Recommended then yield badge okBrush "recommended"
                        if not m.Tested then yield badge mutedBrush "untested"
                        if m.Experimental then yield badge warnBrush "experimental"
                        match m.Fits with
                        | "low-ram" -> yield badge warnBrush "low-RAM mode"
                        | "budget" -> yield badge warnBrush "part from the SSD"
                        | "slow" | "no" -> yield badge failBrush "does not fit the RAM"
                        | _ -> ()
                    ]
                    muted (sprintf "%s; download %s, needs ~%s of RAM" m.About (gb m.DownloadGb) (gb m.RamGb))
                ])
        let contexts = ComboBox(MinWidth = 160.0)
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let gpuBox = ComboBox(ItemsSource = [ for g in gpus -> sprintf "%s (%s)" g.Name (gb g.VramGb) ], MinWidth = 260.0)
        gpuBox.SelectedIndex <- (if d.Gpu.HasValue then max 0 (Array.findIndex (fun g -> g.Index = d.Gpu.Value) gpus) else 0)
        let dataDir = TextBox(Text = d.DataDir, MinWidth = 420.0)
        let ggufDir = TextBox(PlaceholderText = "(optional) a folder with the model files already downloaded", MinWidth = 420.0)
        let license = muted ""
        let install = accentButton "Install" ignore
        let selectedModel () =
            match sizes.SelectedItem with
            | :? Model as m -> Some m
            | _ -> None
        let familyChanged () =
            match families.SelectedItem with
            | :? Family as f ->
                sizes.ItemsSource <- f.Models
                sizes.SelectedItem <- (f.Models |> Array.tryFind (fun m -> m.Recommended) |> Option.defaultValue f.Models[0])
                license.Text <- (if String.IsNullOrEmpty f.License then "" else "Its license: " + f.License)
            | _ -> ()
        let sizeChanged () =
            match families.SelectedItem, selectedModel () with
            | (:? Family as f), Some m ->
                contexts.ItemsSource <- [ for c in f.Contexts -> contextLabel c ]
                contexts.SelectedIndex <- max 0 (Array.IndexOf(f.Contexts, m.Context))
                install.IsEnabled <- true
            | _ -> install.IsEnabled <- false
        families.SelectionChanged.Add(fun _ -> familyChanged ())
        sizes.SelectionChanged.Add(fun _ -> sizeChanged ())
        install.Click.Add(fun _ ->
            match families.SelectedItem, selectedModel () with
            | (:? Family as f), Some m ->
                let args =
                    [ "--setup"; "--yes"; "--no-start"; "--family"; f.Id; "--model"; m.Id
                      "--context"; string f.Contexts[max 0 contexts.SelectedIndex]
                      "--data-dir"; dataDir.Text ]
                    @ (if gpus.Length > 0 then [ "--gpu"; string gpus[max 0 gpuBox.SelectedIndex].Index ] else [])
                    @ (if String.IsNullOrWhiteSpace ggufDir.Text then [] else [ "--gguf-dir"; ggufDir.Text ])
                installPage (f.Title + " " + m.Id) args
            | _ -> ())
        families.SelectedIndex <- 0
        show (
            stack 14.0 [
                heading "Install a model"
                field "Model" families
                field "Size" sizes
                license
                field "Context: how much text the model sees at once (longer takes more VRAM)" contexts
                if gpus.Length > 1 then field "GPU" gpuBox
                field "Data folder: where the model files go (tens of GB)" (row 8.0 [ dataDir; button "Browse ..." (fun () -> pickFolder "Data folder" dataDir) ])
                field "Model files you already have" (row 8.0 [ ggufDir; button "Browse ..." (fun () -> pickFolder "Model files" ggufDir) ])
                actions [ button "Back" (fun () -> homePage d); install ]
            ]
        )

    and installPage (what: string) (args: string list) =
        runPage ("Installing " + what) args (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    and startModel (index: int) (m: Installed) =
        let stopButton = button "Stop" ignore
        let page (url: string) =
            show (
                stack 14.0 [
                    heading m.ModelName
                    card (
                        stack 8.0 [
                            coloured okBrush "Running"
                            text ("Chat in the browser: " + url + "/")
                            text ("API (OpenAI): " + url + "/v1   (Anthropic: " + url + "/v1/messages)")
                        ]
                    )
                    actions [ stopButton; accentButton "Open chat" (fun () -> openUrl (url + "/")) ]
                ]
            )
        // several installed models: setup asks which one, in describe's order (the newest first)
        let which (q: string) = if q.StartsWith "Which one" then Some(string (index + 1)) else None
        let r =
            runPage ("Starting " + m.ModelName) [ "--no-browser" ] which (fun url -> page url) (fun _ ->
                running <- None
                home ())
        running <- Some r
        stopButton.Click.Add(fun _ -> r.Stop())

    do
        let header =
            let g = Grid(ColumnDefinitions = ColumnDefinitions "*,Auto")
            let title = heading "XeStrata"
            version.VerticalAlignment <- VerticalAlignment.Center
            Grid.SetColumn(version, 1)
            g.Children.Add title
            g.Children.Add version
            g
        let top = stack 10.0 [ header; updateBar ]
        top.Margin <- Thickness(24.0, 16.0, 24.0, 0.0)
        let dock = DockPanel()
        DockPanel.SetDock(top, Dock.Top)
        dock.Children.Add top
        dock.Children.Add host
        this.Content <- dock
        this.Closing.Add(fun _ -> running |> Option.iter (fun r -> r.Stop()))
        home ()
        checkUpdate ()
