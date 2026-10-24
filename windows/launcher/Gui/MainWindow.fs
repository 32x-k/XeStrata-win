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
    /// setup.py (or `script`, Setup.Run's) with `args`, its steps, progress, questions and log on a page of its own:
    /// `answer` answers a question the window has already asked (else the page asks it), `onReady` gets the server's
    /// address once it says so, `onDone` the exit code (after a failure: when the page is left).
    let runScriptPage (script: string) (title: string) (args: string list) (answer: string -> string option)
                      (onReady: string -> unit) (onDone: int -> unit) =
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
        let r = Setup.Run(script, args, onMessage, (fun code -> ui (fun () -> if not failed then onDone code)))
        run <- Some r
        r

    let runPage = runScriptPage Places.setupPy

    /// The image models to choose from (describe's), `current` checked (--images names: the default first) -> the
    /// control and its --images value (None: none checked).  A choice of the default where several are, and of the
    /// text encoder with the refusal direction removed (-uncensored) where the models have one.
    let imagePicker (models: ImageModel[]) (current: string[]) =
        let plain (id: string) = if id.EndsWith "-uncensored" then id.Substring(0, id.Length - 11) else id
        let had = current |> Array.map plain
        let boxes =
            [| for m in models ->
                   CheckBox(
                       IsChecked = Nullable(Array.contains m.Id had),
                       Content =
                           stack 2.0 [
                               row 6.0 [
                                   yield subheading m.Title
                                   if not m.Edits then yield badge mutedBrush "no edits"
                               ]
                               muted (sprintf "%s; download %s, needs ~%s of RAM besides the text model" m.About (gb m.DownloadGb) (gb m.RamGb))
                           ]
                   ) |]
        let uncensored =
            CheckBox(
                Content = text "Text encoder with its refusal direction removed (-uncensored)",
                IsChecked = Nullable(current |> Array.exists (fun c -> c.EndsWith "-uncensored"))
            )
        let defaultBox = ComboBox(MinWidth = 260.0)
        let defaultField = field "Default image model (the others: chosen on the web page or by the API's \"model\")" defaultBox
        let checkedModels () = [ for i, b in Array.indexed boxes do if b.IsChecked = Nullable true then models[i] ]
        let refresh () =
            let ms = checkedModels ()
            let was = match defaultBox.SelectedItem with :? string as t -> t | _ -> ""
            let first = if had.Length > 0 then (models |> Array.tryFind (fun m -> m.Id = had[0])) else None
            defaultBox.ItemsSource <- [ for m in ms -> m.Title ]
            defaultBox.SelectedItem <-
                (match ms |> List.tryFind (fun m -> m.Title = was), first with
                 | Some m, _ -> m.Title
                 | None, Some f when List.contains f ms -> f.Title
                 | _ -> (match ms with m :: _ -> m.Title | [] -> null))
            defaultField.IsVisible <- ms.Length > 1
            uncensored.IsVisible <- ms |> List.exists (fun m -> m.Uncensored)
        for b in boxes do
            b.IsCheckedChanged.Add(fun _ -> refresh ())
        refresh ()
        let value () =
            match checkedModels () with
            | [] -> None
            | ms ->
                let def = match defaultBox.SelectedItem with :? string as t -> t | _ -> ms.Head.Title
                let ordered = (ms |> List.filter (fun m -> m.Title = def)) @ (ms |> List.filter (fun m -> m.Title <> def))
                let name (m: ImageModel) =
                    if uncensored.IsChecked = Nullable true && m.Uncensored then m.Id + "-uncensored" else m.Id
                Some(ordered |> List.map name |> String.concat ",")
        let control =
            stack 8.0 [
                yield! (boxes |> Array.map (fun b -> b :> Control))
                yield defaultField :> Control
                yield uncensored :> Control
            ]
        control, value

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
                                  + (if m.Images.Length > 0 then ", images: " + String.Join(", ", m.Images) else "")
                              )
                          ]
                      let imagesOk = d.Families |> Array.exists (fun f -> f.Id = m.Family && f.Images)
                      let buttons =
                          row 8.0 [
                              if imagesOk && d.Images.Length > 0 then
                                  yield button "Images ..." (fun () -> imagesPage d m)
                              yield accentButton "Start" (fun () -> startModel i m)
                          ]
                      Grid.SetColumn(buttons, 1)
                      buttons.VerticalAlignment <- VerticalAlignment.Center
                      g.Children.Add info
                      g.Children.Add buttons
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
        let images, imagesValue = imagePicker d.Images [||]
        let imagesField = field "Image generation (optional): models to make images with, besides the text model" images
        let selectedModel () =
            match sizes.SelectedItem with
            | :? Model as m -> Some m
            | _ -> None
        let familyChanged () =
            match families.SelectedItem with
            | :? Family as f ->
                sizes.ItemsSource <- f.Models
                sizes.SelectedItem <- (f.Models |> Array.tryFind (fun m -> m.Recommended) |> Option.defaultValue f.Models[0])
                imagesField.IsVisible <- f.Images && d.Images.Length > 0
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
                    @ (match imagesValue () with
                       | Some v when f.Images -> [ "--images"; v ]
                       | _ -> [])
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
                imagesField
                actions [ button "Back" (fun () -> homePage d); install ]
            ]
        )

    and installPage (what: string) (args: string list) =
        runPage ("Installing " + what) args (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    /// An installed model's image generation: its image models chosen again (none: taken out), the files downloaded
    /// and its config rewritten by tools/image/install.py; the server takes it at its next start.
    and imagesPage (d: Describe) (m: Installed) =
        let picker, value = imagePicker d.Images m.Images
        let apply () =
            let v = value () |> Option.defaultValue "no"
            runScriptPage Places.imageInstallPy ("Image generation for " + m.ModelName)
                [ "--config"; m.Config; "--model"; v; "--yes" ] (fun _ -> None) ignore (fun _ -> home ())
            |> ignore
        show (
            stack 14.0 [
                heading ("Image generation for " + m.ModelName)
                muted "Check the image models to make images with (none: image generation off). Their files are downloaded, and the model takes them at its next start."
                picker
                actions [ button "Back" (fun () -> homePage d); accentButton "Apply" apply ]
            ]
        )

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
