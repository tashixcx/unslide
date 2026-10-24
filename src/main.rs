use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use lopdf::{Document, Object, ObjectId};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "unslide",
    version,
    about = "Ultra-fast, lossless tool to detect and collapse progressive build/animation slides in PDFs."
)]
struct Cli {
    /// PDF file(s) to process.
    /// - 1 file: saves to <name>_clean.pdf (or specify -o / -i)
    /// - 2+ files: batch processes all files (saves to <name>_clean.pdf, -d, or -i)
    #[arg(required = true, value_name = "FILES")]
    files: Vec<PathBuf>,

    /// Destination output file (for single file processing only)
    #[arg(short = 'o', long = "output", value_name = "OUTPUT_FILE", conflicts_with = "in_place")]
    output: Option<PathBuf>,

    /// Output directory for batch processing
    #[arg(short = 'd', long = "output-dir", value_name = "DIR", conflicts_with = "in_place")]
    output_dir: Option<PathBuf>,

    /// Overwrite the original input file in-place (atomic and safe)
    #[arg(short = 'i', long = "in-place", conflicts_with_all = ["output", "output_dir"])]
    in_place: bool,

    /// Skip unreferenced object pruning (faster write, but output PDF file size will not shrink)
    #[arg(long = "no-compact", alias = "fast")]
    no_compact: bool,

    /// Detection strategy
    #[arg(short, long, value_enum, default_value_t = Mode::Auto)]
    mode: Mode,

    /// Containment similarity threshold for text mode (0.50 - 1.0)
    #[arg(short, long, default_value_t = 0.85)]
    threshold: f64,

    /// Output machine-readable JSON summary to stdout
    #[arg(long)]
    json: bool,

    /// Suppress informational output (errors and JSON still printed)
    #[arg(short, long)]
    quiet: bool,

    /// Preview which pages will be removed without writing to disk
    #[arg(short = 'n', long = "dry-run")]
    dry_run: bool,

    /// Force overwrite of existing destination files
    #[arg(short, long)]
    force: bool,

    /// Print detailed detection logs per slide
    #[arg(short, long)]
    verbose: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// Auto-detect: Uses LaTeX Beamer frame labels to enforce frame boundaries, validated by content
    Auto,
    /// Force LaTeX Beamer metadata validation
    Beamer,
    /// Force pure text and structural analysis (ignores Beamer metadata)
    Text,
}

struct Job {
    input: PathBuf,
    output: PathBuf,
    in_place: bool,
}

#[derive(Serialize)]
struct JobReport {
    input: String,
    output: String,
    engine: Mode,
    original_pages: usize,
    kept_pages: usize,
    removed_pages: usize,
    reduction_percent: f64,
    kept: Vec<u32>,
    removed: Vec<u32>,
}

#[derive(Clone, Debug)]
struct SlideInfo {
    page_num: u32,
    title_tokens: Vec<String>,
    body_tokens: Vec<String>,
    tokens: Vec<String>,
    token_set: HashSet<String>,
    xobjects: HashSet<ObjectId>,
    content_len: usize,
    mediabox: Option<[f64; 4]>,
    rotation: i64,
    frame_id: Option<i64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let jobs = plan_jobs(&cli)?;

    if let Some(ref dir) = cli.output_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
    }

    let mut reports = Vec::new();
    let mut failed = 0;

    for job in &jobs {
        match process_job(job, &cli) {
            Ok(report) => reports.push(report),
            Err(err) => {
                eprintln!("\n[ERROR] Failed to process '{}': {:#}", job.input.display(), err);
                failed += 1;
            }
        }
    }

    if cli.json {
        let json_str = serde_json::to_string_pretty(&reports)
            .context("Failed to serialize JSON report")?;
        println!("{}", json_str);
    }

    if failed > 0 {
        bail!("Completed with {} error(s).", failed);
    }

    Ok(())
}

fn plan_jobs(cli: &Cli) -> Result<Vec<Job>> {
    let files = &cli.files;

    if files.len() == 1 {
        let input = files[0].clone();
        let in_place = cli.in_place;
        let output = if in_place {
            input.clone()
        } else if let Some(ref o) = cli.output {
            o.clone()
        } else if let Some(ref dir) = cli.output_dir {
            let name = input.file_name().context("Invalid input filename")?;
            dir.join(name)
        } else {
            derive_default_output(&input)
        };
        return Ok(vec![Job { input, output, in_place }]);
    }

    if cli.output.is_some() {
        bail!("Cannot specify a single --output (-o) with multiple input files. Use --output-dir (-d) or --in-place (-i).");
    }

    let mut jobs = Vec::new();
    for input in files {
        let in_place = cli.in_place;
        let output = if in_place {
            input.clone()
        } else if let Some(ref dir) = cli.output_dir {
            let name = input.file_name().context("Invalid input filename")?;
            dir.join(name)
        } else {
            derive_default_output(input)
        };
        jobs.push(Job { input: input.clone(), output, in_place });
    }

    Ok(jobs)
}

fn process_job(job: &Job, cli: &Cli) -> Result<JobReport> {
    if !job.input.is_file() {
        bail!("File not found or is not a regular file: {}", job.input.display());
    }

    if job.output.exists() && !cli.force && !job.in_place && !cli.dry_run {
        bail!("Output file '{}' already exists. Use --force (-f) or --in-place (-i).", job.output.display());
    }

    if !cli.quiet && !cli.json {
        println!("\nLoading: {}", job.input.display());
    }

    let mut doc = Document::load(&job.input)
        .with_context(|| format!("Failed to parse PDF: {}", job.input.display()))?;

    let page_map = doc.get_pages();
    let total_pages = page_map.len();

    if total_pages <= 1 {
        if !cli.quiet && !cli.json {
            println!("PDF contains only {} page. Skipping.", total_pages);
        }
        return Ok(JobReport {
            input: job.input.display().to_string(),
            output: job.output.display().to_string(),
            engine: cli.mode,
            original_pages: total_pages,
            kept_pages: total_pages,
            removed_pages: 0,
            reduction_percent: 0.0,
            kept: (1..=total_pages as u32).collect(),
            removed: Vec::new(),
        });
    }

    let beamer_frames = if matches!(cli.mode, Mode::Auto | Mode::Beamer) {
        extract_beamer_frame_info(&doc, total_pages)
    } else {
        None
    };

    if cli.mode == Mode::Beamer && beamer_frames.is_none() {
        bail!("Beamer mode explicitly requested, but no valid Beamer PageLabels structure was found.");
    }

    let detected_mode = if beamer_frames.is_some() {
        Mode::Beamer
    } else {
        Mode::Text
    };

    if !cli.quiet && !cli.json {
        println!(
            "Analyzing slide structures, layout, and content across CPU cores (engine: {:?})...",
            detected_mode
        );
    }

    let keep = detect_content_groups(
        &doc,
        &page_map,
        total_pages,
        beamer_frames,
        cli.threshold,
        cli.verbose,
    );

    let removed_count = total_pages.saturating_sub(keep.len());
    let reduction = (removed_count as f64 / total_pages as f64) * 100.0;

    let all_pages: Vec<u32> = (1..=total_pages as u32).collect();
    let removed_pages: Vec<u32> = all_pages.into_iter().filter(|p| !keep.contains(p)).collect();

    if !cli.quiet && !cli.json {
        println!("=== Summary ===");
        println!("  Engine:             {:?}", detected_mode);
        println!("  Original:           {}", total_pages);
        println!("  Builds removed:     {}", removed_count);
        println!("  Clean slides kept:  {}", keep.len());
        println!("  Reduction:          {:.1}%", reduction);

        if (cli.verbose || cli.dry_run) && removed_count > 0 {
            println!("Removed pages: {:?}", removed_pages);
            println!("Kept pages:    {:?}", keep);
        }
    }

    let report = JobReport {
        input: job.input.display().to_string(),
        output: job.output.display().to_string(),
        engine: detected_mode,
        original_pages: total_pages,
        kept_pages: keep.len(),
        removed_pages: removed_count,
        reduction_percent: reduction,
        kept: keep.clone(),
        removed: removed_pages,
    };

    if removed_count == 0 {
        if !cli.quiet && !cli.json {
            println!("No progressive slides detected. Document is already clean.");
        }
        return Ok(report);
    }

    if cli.dry_run {
        if !cli.quiet && !cli.json {
            println!("[DRY RUN] No modifications made.");
        }
        return Ok(report);
    }

    remap_outlines_to_kept(&mut doc, &page_map, &keep);

    if !cli.quiet && !cli.json {
        println!("Collapsing progressive slides...");
    }
    fast_retain_pages(&mut doc, &page_map, &keep)?;

    if !cli.no_compact && removed_count > 0 {
        if !cli.quiet && !cli.json {
            println!("Pruning unreferenced objects and orphaned streams...");
        }
        doc.prune_objects();
    }

    if job.in_place {
        let temp_name = format!(
            "{}.{}.unslide.tmp",
            job.input.file_name().and_then(|s| s.to_str()).unwrap_or("file"),
            std::process::id()
        );
        let temp_path = job.input.with_file_name(temp_name);

        let save_res = doc.save(&temp_path);
        if let Err(e) = save_res {
            let _ = fs::remove_file(&temp_path);
            return Err(e).with_context(|| format!("Failed to write temporary file: {}", temp_path.display()));
        }

        if let Err(e) = fs::rename(&temp_path, &job.input) {
            let _ = fs::remove_file(&temp_path);
            return Err(e).with_context(|| format!("Failed to safely overwrite: {}", job.input.display()));
        }

        if !cli.quiet && !cli.json {
            println!("Updated in-place: {}", job.input.display());
        }
    } else {
        if !cli.quiet && !cli.json {
            println!("Saving to: {}", job.output.display());
        }
        doc.save(&job.output)
            .with_context(|| format!("Failed to save output to: {}", job.output.display()))?;
    }

    if !cli.quiet && !cli.json {
        println!("Done!");
    }

    Ok(report)
}

fn fast_retain_pages(
    doc: &mut Document,
    page_map: &BTreeMap<u32, ObjectId>,
    keep: &[u32],
) -> Result<()> {
    let pages_id = match get_root_pages_id(doc) {
        Some(id) => id,
        None => {
            let all_pages: Vec<u32> = page_map.keys().copied().collect();
            let to_delete: Vec<u32> = all_pages.into_iter().filter(|p| !keep.contains(p)).collect();
            doc.delete_pages(&to_delete);
            return Ok(());
        }
    };

    let kept_page_ids: Vec<ObjectId> = keep
        .iter()
        .filter_map(|p| page_map.get(p).copied())
        .collect();

    for &page_id in &kept_page_ids {
        resolve_inherited_attributes(doc, page_id);
    }

    let pages_obj = doc
        .get_object_mut(pages_id)
        .with_context(|| "Failed to get root Pages object")?;
    let pages_dict = pages_obj
        .as_dict_mut()
        .map_err(|e| anyhow::anyhow!("Root Pages is not a dictionary: {:?}", e))?;

    pages_dict.set(b"Count".as_slice(), Object::Integer(kept_page_ids.len() as i64));
    pages_dict.set(
        b"Kids".as_slice(),
        Object::Array(kept_page_ids.iter().map(|&id| Object::Reference(id)).collect()),
    );

    for &id in &kept_page_ids {
        if let Ok(page_dict) = doc.get_object_mut(id).and_then(|o| o.as_dict_mut()) {
            page_dict.set(b"Parent".as_slice(), Object::Reference(pages_id));
        }
    }

    for (page_num, &obj_id) in page_map {
        if !keep.contains(page_num) {
            doc.objects.remove(&obj_id);
        }
    }

    let orphaned_pages_ids: Vec<ObjectId> = doc
        .objects
        .iter()
        .filter_map(|(&id, obj)| {
            if id == pages_id {
                return None;
            }
            if let Object::Dictionary(dict) = obj {
                if let Ok(Object::Name(name)) = dict.get(b"Type".as_slice()) {
                    if name.as_slice() == b"Pages" {
                        return Some(id);
                    }
                }
            }
            None
        })
        .collect();

    for id in orphaned_pages_ids {
        doc.objects.remove(&id);
    }

    if let Ok(catalog) = doc.catalog_mut() {
        catalog.remove(b"StructTreeRoot".as_slice());

        let mut labels_dict = lopdf::Dictionary::new();
        let mut label_spec = lopdf::Dictionary::new();
        label_spec.set(b"S".as_slice(), Object::Name(b"D".to_vec()));
        label_spec.set(b"St".as_slice(), Object::Integer(1));
        labels_dict.set(
            b"Nums".as_slice(),
            Object::Array(vec![Object::Integer(0), Object::Dictionary(label_spec)]),
        );
        catalog.set(b"PageLabels".as_slice(), Object::Dictionary(labels_dict));
    }

    Ok(())
}

fn resolve_inherited_attributes(doc: &mut Document, page_id: ObjectId) {
    let check_attrs = [
        b"MediaBox".as_slice(),
        b"CropBox".as_slice(),
        b"Rotate".as_slice(),
        b"Resources".as_slice(),
    ];

    let mut missing = Vec::new();
    if let Ok(page_dict) = doc.get_object(page_id).and_then(|o| o.as_dict()) {
        for &attr in &check_attrs {
            if page_dict.get(attr).is_err() {
                missing.push(attr);
            }
        }
    }

    if missing.is_empty() {
        return;
    }

    let mut inherited: HashMap<&[u8], Object> = HashMap::new();
    let mut curr_id = page_id;
    let mut visited = HashSet::new();

    while let Ok(dict) = doc.get_object(curr_id).and_then(|o| o.as_dict()) {
        for &attr in &missing {
            if !inherited.contains_key(attr) {
                if let Ok(val) = dict.get(attr) {
                    inherited.insert(attr, val.clone());
                }
            }
        }
        if inherited.len() == missing.len() {
            break;
        }
        if let Ok(parent_ref) = dict.get(b"Parent".as_slice()).and_then(|p| p.as_reference()) {
            if !visited.insert(parent_ref) || visited.len() > 32 {
                break;
            }
            curr_id = parent_ref;
        } else {
            break;
        }
    }

    if let Ok(page_dict) = doc.get_object_mut(page_id).and_then(|o| o.as_dict_mut()) {
        for (attr, val) in inherited {
            page_dict.set(attr, val);
        }
    }
}

fn get_root_pages_id(doc: &Document) -> Option<ObjectId> {
    if let Ok(catalog) = doc.catalog() {
        if let Ok(pages_ref) = catalog.get(b"Pages".as_slice()).and_then(|o| o.as_reference()) {
            return Some(pages_ref);
        }
    }
    let root_ref = doc.trailer.get(b"Root".as_slice()).ok()?.as_reference().ok()?;
    let root_obj = doc.get_object(root_ref).ok()?;
    let root_dict = match root_obj {
        Object::Dictionary(d) => d,
        _ => return None,
    };
    root_dict.get(b"Pages".as_slice()).ok()?.as_reference().ok()
}

fn derive_default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("document");
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("pdf");
    input.with_file_name(format!("{}_clean.{}", stem, ext))
}

fn extract_beamer_frame_info(doc: &Document, total_pages: usize) -> Option<Vec<Option<i64>>> {
    let catalog = doc.catalog().ok().or_else(|| {
        let root = doc.trailer.get(b"Root".as_slice()).ok()?;
        let (_, obj) = doc.dereference(root).ok()?;
        match obj {
            Object::Dictionary(d) => Some(d),
            _ => None,
        }
    })?;

    let page_labels_obj = catalog.get(b"PageLabels".as_slice()).ok()?;
    let (_, labels_resolved) = doc.dereference(page_labels_obj).ok()?;
    let labels_dict = match labels_resolved {
        Object::Dictionary(d) => d,
        _ => return None,
    };

    let nums_arr = get_nums_array(doc, labels_dict)?;
    let mut entries: Vec<(usize, i64)> = Vec::new();

    for i in (0..nums_arr.len()).step_by(2) {
        if let Object::Integer(k) = nums_arr[i] {
            if let Some(dict_obj) = nums_arr.get(i + 1) {
                if let Ok((_, Object::Dictionary(d))) = doc.dereference(dict_obj) {
                    let mut num = None;
                    if let Ok(Object::Integer(st)) = d.get(b"St".as_slice()) {
                        num = Some(*st);
                    } else if let Ok(s) = d.get(b"P".as_slice()).and_then(|o| o.as_str()) {
                        let s_str = String::from_utf8_lossy(s);
                        let digits: String = s_str.chars().filter(|c| c.is_ascii_digit()).collect();
                        if let Ok(parsed) = digits.parse::<i64>() {
                            num = Some(parsed);
                        }
                    }
                    if let Some(n) = num {
                        entries.push((k as usize, n));
                    }
                }
            }
        }
    }

    if entries.is_empty() || entries[0].0 != 0 {
        return None;
    }

    entries.sort_by_key(|&(k, _)| k);

    let mut page_frames = vec![None; total_pages];
    for (idx, &(k, frame_num)) in entries.iter().enumerate() {
        let next_k = if idx + 1 < entries.len() {
            entries[idx + 1].0
        } else {
            total_pages
        };
        for p in k..next_k.min(total_pages) {
            page_frames[p] = Some(frame_num);
        }
    }

    Some(page_frames)
}

fn get_nums_array(doc: &Document, dict: &lopdf::Dictionary) -> Option<Vec<Object>> {
    if let Ok(nums_obj) = dict.get(b"Nums".as_slice()) {
        if let Ok((_, resolved)) = doc.dereference(nums_obj) {
            if let Object::Array(arr) = resolved {
                return Some(arr.clone());
            }
        }
    }
    if let Ok(kids_obj) = dict.get(b"Kids".as_slice()) {
        if let Ok((_, resolved)) = doc.dereference(kids_obj) {
            if let Object::Array(kids) = resolved {
                let mut all = Vec::new();
                for kid in kids {
                    if let Ok((_, kid_res)) = doc.dereference(kid) {
                        if let Object::Dictionary(kid_dict) = kid_res {
                            if let Some(sub) = get_nums_array(doc, kid_dict) {
                                all.extend(sub);
                            }
                        }
                    }
                }
                if !all.is_empty() {
                    return Some(all);
                }
            }
        }
    }
    None
}

fn detect_content_groups(
    doc: &Document,
    page_map: &BTreeMap<u32, ObjectId>,
    total_pages: usize,
    beamer_frames: Option<Vec<Option<i64>>>,
    threshold: f64,
    verbose: bool,
) -> Vec<u32> {
    struct RawSlide {
        page_num: u32,
        raw_text: String,
        tokens: Vec<String>,
        token_set: HashSet<String>,
        xobjects: HashSet<ObjectId>,
        content_len: usize,
        mediabox: Option<[f64; 4]>,
        rotation: i64,
        frame_id: Option<i64>,
    }

    let raw_slides: Vec<RawSlide> = (1..=total_pages as u32)
        .into_par_iter()
        .map(|page_num| {
            let raw_text = doc.extract_text(&[page_num]).unwrap_or_default();
            let page_str = page_num.to_string();
            let tokens = extract_slide_words(&raw_text, &page_str);
            let token_set: HashSet<String> = tokens.iter().cloned().collect();

            let mut xobjects = HashSet::new();
            let mut content_len = 0;
            let mut mediabox = None;
            let mut rotation = 0;

            if let Some(&page_id) = page_map.get(&page_num) {
                mediabox = extract_mediabox(doc, page_id);
                content_len = extract_content_len(doc, page_id);
                xobjects = extract_xobjects(doc, page_id);
                rotation = extract_rotation(doc, page_id);
            }

            let frame_id = beamer_frames
                .as_ref()
                .and_then(|f| f.get((page_num - 1) as usize).copied().flatten());

            RawSlide {
                page_num,
                raw_text,
                tokens,
                token_set,
                xobjects,
                content_len,
                mediabox,
                rotation,
                frame_id,
            }
        })
        .collect();

    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for slide in &raw_slides {
        for t in &slide.token_set {
            *doc_freq.entry(t.clone()).or_insert(0) += 1;
        }
    }

    let boilerplate_cutoff = (total_pages as f64 * 0.80).ceil() as usize;
    let boilerplate: HashSet<String> = doc_freq
        .into_iter()
        .filter(|&(_, count)| total_pages >= 5 && count >= boilerplate_cutoff)
        .map(|(t, _)| t)
        .collect();

    let slides: Vec<SlideInfo> = raw_slides
        .into_iter()
        .map(|raw| {
            let page_str = raw.page_num.to_string();
            let (title_tokens, body_tokens) = extract_title_and_body(&raw.raw_text, &boilerplate, &page_str);

            SlideInfo {
                page_num: raw.page_num,
                title_tokens,
                body_tokens,
                tokens: raw.tokens,
                token_set: raw.token_set,
                xobjects: raw.xobjects,
                content_len: raw.content_len,
                mediabox: raw.mediabox,
                rotation: raw.rotation,
                frame_id: raw.frame_id,
            }
        })
        .collect();

    let mut keep_pages = Vec::new();

    for i in 0..slides.len() - 1 {
        let curr = &slides[i];
        let next = &slides[i + 1];

        if is_progressive_build(curr, next, threshold) {
            if verbose {
                println!(
                    "  [Build] Page {} collapses into Page {}",
                    curr.page_num, next.page_num
                );
            }
        } else {
            keep_pages.push(curr.page_num);
        }
    }

    if let Some(last) = slides.last() {
        keep_pages.push(last.page_num);
    }

    keep_pages.dedup();
    keep_pages.sort_unstable();
    keep_pages
}

fn is_progressive_build(a: &SlideInfo, b: &SlideInfo, threshold: f64) -> bool {
    // 1. Structural Beamer Frame Check:
    // If Beamer metadata explicitly assigns different frame counters, they can NEVER collapse!
    if let (Some(fa), Some(fb)) = (a.frame_id, b.frame_id) {
        if fa != fb {
            return false;
        }
    }

    // 2. Rotation Check
    if a.rotation != b.rotation {
        return false;
    }

    // 3. Geometry Check
    if let (Some(box_a), Some(box_b)) = (a.mediabox, b.mediabox) {
        let width_a = (box_a[2] - box_a[0]).abs();
        let height_a = (box_a[3] - box_a[1]).abs();
        let width_b = (box_b[2] - box_b[0]).abs();
        let height_b = (box_b[3] - box_b[1]).abs();

        if (width_a - width_b).abs() > 2.0 || (height_a - height_b).abs() > 2.0 {
            return false;
        }
    }

    // 4. Title Overlap Check
    let title_overlap = if !a.title_tokens.is_empty() && !b.title_tokens.is_empty() {
        a.title_tokens.iter().filter(|t| b.title_tokens.contains(t)).count()
    } else {
        0
    };

    if !a.title_tokens.is_empty() && !b.title_tokens.is_empty() && title_overlap == 0 {
        return false;
    }

    // 5. SECTION DIVIDER / TITLE SLIDE GUARD:
    // A slide containing ONLY a title and NO body content (like "Bootloader" or "Armado de GDT")
    // followed by a slide with body content is a section title slide, NEVER an overlay build!
    if a.body_tokens.is_empty() && !b.body_tokens.is_empty() {
        return false;
    }

    // 6. Image Replacement Guard: If images were removed or replaced, this is a distinct slide
    if !a.xobjects.is_empty() && !a.xobjects.is_subset(&b.xobjects) {
        return false;
    }

    // 7. Visual Diagram Build Check (both slides have zero body text)
    if a.body_tokens.is_empty() && b.body_tokens.is_empty() {
        let has_matching_title = !a.title_tokens.is_empty() && a.title_tokens == b.title_tokens;
        let has_strict_image_addition = b.xobjects.len() > a.xobjects.len() && a.xobjects.is_subset(&b.xobjects);

        if a.xobjects.is_subset(&b.xobjects) {
            if (has_matching_title && a.content_len > 0 && b.content_len > a.content_len) || has_strict_image_addition {
                return true;
            }
        }
        return false;
    }

    let a_tokens = &a.body_tokens;
    let b_tokens = &b.body_tokens;

    // 8. NEAR-DUPLICATE & HIGHLIGHT / ALERT OVERLAY COLLAPSE:
    // If both slides share the same title, same frame, and have virtually identical body content
    // (e.g. green highlight on a code keyword, alert color change, or trailing \pause):
    // Slide A is an unhighlighted intermediate state and MUST collapse into Slide B!
    if !a_tokens.is_empty() && !b_tokens.is_empty() {
        let (matched_forward, b_idx) = count_subsequence_matches(a_tokens, b_tokens);
        let order_ratio = matched_forward as f64 / a_tokens.len() as f64;
        let b_ratio = matched_forward as f64 / b_tokens.len() as f64;

        // If >= 85% of words match in both directions, it is the same slide with a visual highlight/color change
        if order_ratio >= 0.85 && b_ratio >= 0.85 {
            return true;
        }

        // 9. PROGRESSIVE EXPANSION MATCHING:
        // Slide B strictly builds upon Slide A, allowing up to 15% extraction noise tolerance
        let min_allowed_len = (a_tokens.len() as f64 * 0.85).floor() as usize;
        let max_allowed_len = a_tokens.len() * 4;

        if b_tokens.len() >= min_allowed_len && (b_tokens.len() <= max_allowed_len || a_tokens.len() < 6) {
            let max_allowed_span = (a_tokens.len() as f64 * 2.2).ceil() as usize;
            if (b_idx <= max_allowed_span || a_tokens.len() < 6) && order_ratio >= threshold {
                return true;
            }

            // Secondary Multiset Containment Fallback (PowerPoint/Keynote out-of-order text frames)
            if title_overlap > 0 && a_tokens.len() >= 4 {
                let mut b_freq: HashMap<&str, usize> = HashMap::new();
                for token in b_tokens {
                    *b_freq.entry(token.as_str()).or_insert(0) += 1;
                }
                let mut matched_multiset = 0;
                for token in a_tokens {
                    if let Some(count) = b_freq.get_mut(token.as_str()) {
                        if *count > 0 {
                            matched_multiset += 1;
                            *count -= 1;
                        }
                    }
                }
                let containment_ratio = matched_multiset as f64 / a_tokens.len() as f64;
                if containment_ratio >= threshold.max(0.90) {
                    return true;
                }
            }
        }
    }

    false
}

fn count_subsequence_matches(a_tokens: &[String], b_tokens: &[String]) -> (usize, usize) {
    let mut b_idx = 0;
    let mut matched = 0;
    for token in a_tokens {
        if let Some(pos) = b_tokens[b_idx..].iter().position(|t| t == token) {
            matched += 1;
            b_idx += pos + 1;
        }
    }
    (matched, b_idx)
}

fn extract_title_and_body(
    raw: &str,
    boilerplate: &HashSet<String>,
    page_str: &str,
) -> (Vec<String>, Vec<String>) {
    let normalized = normalize_typography(raw);
    let mut title = Vec::new();
    let mut body = Vec::new();
    let mut found_title = false;

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_slide_number_line(trimmed) {
            continue;
        }
        let words = extract_words(trimmed, page_str);
        if words.is_empty() {
            continue;
        }

        if !found_title {
            if !words.iter().all(|w| boilerplate.contains(w)) {
                title = words;
                found_title = true;
            }
        } else {
            let non_bp: Vec<String> = words
                .into_iter()
                .filter(|w| !boilerplate.contains(w))
                .collect();
            body.extend(non_bp);
        }
    }

    (title, body)
}

fn extract_slide_words(raw: &str, page_str: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let normalized = normalize_typography(raw);

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_slide_number_line(trimmed) {
            continue;
        }
        tokens.extend(extract_words(trimmed, page_str));
    }
    tokens
}

fn extract_words(line: &str, page_str: &str) -> Vec<String> {
    let page_clean = page_str.trim_start_matches('0');
    line.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| {
            let clean = w.trim_start_matches('0');
            clean != page_clean && !clean.is_empty()
        })
        .collect()
}

fn normalize_typography(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{FB05}' => out.push_str("ft"),
            '\u{FB06}' => out.push_str("st"),
            '\u{2018}'..='\u{201B}' => out.push('\''),
            '\u{201C}'..='\u{201F}' => out.push('"'),
            '\u{2013}' | '\u{2014}' | '\u{2212}' => out.push('-'),
            '\u{00A0}' | '\u{202F}' | '\u{2007}' => out.push(' '),
            '\u{200B}'..='\u{200D}' | '\u{FEFF}' => {}
            _ => out.push(c),
        }
    }
    out
}

fn is_slide_number_line(line: &str) -> bool {
    let s = line.trim().to_lowercase();
    if s.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if let Some((left, right)) = s.split_once('/') {
        if !left.trim().is_empty()
            && !right.trim().is_empty()
            && left.trim().chars().all(|c| c.is_ascii_digit())
            && right.trim().chars().all(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    if s.starts_with("slide ") || s.starts_with("page ") {
        let rest = s.trim_start_matches("slide ").trim_start_matches("page ").trim();
        if rest.chars().all(|c| c.is_ascii_digit() || c.is_whitespace() || c == '/') {
            return true;
        }
    }
    false
}

fn extract_mediabox(doc: &Document, page_id: ObjectId) -> Option<[f64; 4]> {
    let mut curr = page_id;
    let mut visited = HashSet::new();

    while let Ok(dict) = doc.get_object(curr).and_then(|o| o.as_dict()) {
        if let Ok(arr) = dict.get(b"MediaBox".as_slice()).and_then(|o| o.as_array()) {
            if arr.len() == 4 {
                let mut vals = [0.0; 4];
                let mut valid = true;
                for (idx, obj) in arr.iter().enumerate() {
                    match obj {
                        Object::Real(f) => vals[idx] = *f as f64,
                        Object::Integer(i) => vals[idx] = *i as f64,
                        _ => valid = false,
                    }
                }
                if valid {
                    return Some(vals);
                }
            }
        }
        if let Ok(parent_ref) = dict.get(b"Parent".as_slice()).and_then(|p| p.as_reference()) {
            if !visited.insert(parent_ref) || visited.len() > 32 {
                break;
            }
            curr = parent_ref;
        } else {
            break;
        }
    }
    None
}

fn extract_rotation(doc: &Document, page_id: ObjectId) -> i64 {
    let mut curr = page_id;
    let mut visited = HashSet::new();

    while let Ok(dict) = doc.get_object(curr).and_then(|o| o.as_dict()) {
        if let Ok(Object::Integer(rot)) = dict.get(b"Rotate".as_slice()) {
            return rot.rem_euclid(360);
        }
        if let Ok(parent_ref) = dict.get(b"Parent".as_slice()).and_then(|p| p.as_reference()) {
            if !visited.insert(parent_ref) || visited.len() > 32 {
                break;
            }
            curr = parent_ref;
        } else {
            break;
        }
    }
    0
}

fn extract_content_len(doc: &Document, page_id: ObjectId) -> usize {
    let mut curr = page_id;
    let mut visited = HashSet::new();

    while let Ok(dict) = doc.get_object(curr).and_then(|o| o.as_dict()) {
        if let Ok(contents) = dict.get(b"Contents".as_slice()) {
            let len = match contents {
                Object::Reference(id) => doc.get_object(*id).map(object_decompressed_len).unwrap_or(0),
                Object::Stream(stream) => stream.decompressed_content().map(|c| c.len()).unwrap_or(stream.content.len()),
                Object::Array(arr) => arr
                    .iter()
                    .filter_map(|o| o.as_reference().ok())
                    .filter_map(|id| doc.get_object(id).ok())
                    .map(object_decompressed_len)
                    .sum(),
                _ => 0,
            };
            if len > 0 {
                return len;
            }
        }
        if let Ok(parent_ref) = dict.get(b"Parent".as_slice()).and_then(|p| p.as_reference()) {
            if !visited.insert(parent_ref) || visited.len() > 32 {
                break;
            }
            curr = parent_ref;
        } else {
            break;
        }
    }
    0
}

fn object_decompressed_len(obj: &Object) -> usize {
    match obj {
        Object::Stream(ref stream) => stream.decompressed_content().map(|c| c.len()).unwrap_or(stream.content.len()),
        _ => 0,
    }
}

fn extract_xobjects(doc: &Document, page_id: ObjectId) -> HashSet<ObjectId> {
    let mut set = HashSet::new();
    let mut curr = page_id;
    let mut visited = HashSet::new();

    while let Ok(dict) = doc.get_object(curr).and_then(|o| o.as_dict()) {
        if let Ok(res_obj) = dict.get(b"Resources".as_slice()) {
            if let Ok((_, Object::Dictionary(ref res))) = doc.dereference(res_obj) {
                if let Ok(xobj_obj) = res.get(b"XObject".as_slice()) {
                    if let Ok((_, Object::Dictionary(ref xobj_dict))) = doc.dereference(xobj_obj) {
                        for (_, val) in xobj_dict.iter() {
                            if let Ok(id) = val.as_reference() {
                                set.insert(id);
                            }
                        }
                        if !set.is_empty() {
                            break;
                        }
                    }
                }
            }
        }
        if let Ok(parent_ref) = dict.get(b"Parent".as_slice()).and_then(|p| p.as_reference()) {
            if !visited.insert(parent_ref) || visited.len() > 32 {
                break;
            }
            curr = parent_ref;
        } else {
            break;
        }
    }
    set
}

fn remap_outlines_to_kept(
    doc: &mut Document,
    page_map: &BTreeMap<u32, ObjectId>,
    keep: &[u32],
) {
    let mut redirect_map: HashMap<ObjectId, ObjectId> = HashMap::new();

    for (&page_num, &obj_id) in page_map {
        if !keep.contains(&page_num) {
            if let Some(&surviving_page) = keep.iter().find(|&&p| p > page_num) {
                if let Some(&surviving_id) = page_map.get(&surviving_page) {
                    redirect_map.insert(obj_id, surviving_id);
                }
            }
        }
    }

    if redirect_map.is_empty() {
        return;
    }

    for (_, object) in doc.objects.iter_mut() {
        remap_object(object, &redirect_map);
    }
}

fn remap_object(obj: &mut Object, redirect_map: &HashMap<ObjectId, ObjectId>) {
    match obj {
        Object::Array(arr) => {
            if is_destination_array(arr) {
                if let Object::Reference(ref mut id) = arr[0] {
                    if let Some(&target) = redirect_map.get(id) {
                        *id = target;
                    }
                }
            } else {
                for item in arr.iter_mut() {
                    remap_object(item, redirect_map);
                }
            }
        }
        Object::Dictionary(dict) => {
            for (_, val) in dict.iter_mut() {
                remap_object(val, redirect_map);
            }
        }
        _ => {}
    }
}

fn is_destination_array(arr: &[Object]) -> bool {
    if arr.len() < 2 {
        return false;
    }
    if !matches!(arr[0], Object::Reference(_)) {
        return false;
    }
    match &arr[1] {
        Object::Name(name) => matches!(
            name.as_slice(),
            b"XYZ" | b"Fit" | b"FitH" | b"FitV" | b"FitR" | b"FitB" | b"FitBH" | b"FitBV"
        ),
        _ => false,
    }
}
