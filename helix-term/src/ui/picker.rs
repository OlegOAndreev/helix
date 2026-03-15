mod handlers;
mod query;

use crate::{
    alt,
    compositor::{self, Component, Compositor, Context, Event, EventResult},
    ctrl, key, shift,
    ui::{
        self,
        document::{render_document, LinePos, TextRenderer},
        picker::query::PickerQuery,
        text_decorations::DecorationManager,
        EditorView,
    },
};
use futures_util::future::BoxFuture;
use helix_event::AsyncHook;
use nucleo::pattern::{CaseMatching, Normalization};
use nucleo::{Config, Nucleo};
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tui::{
    buffer::Buffer as Surface,
    layout::Constraint,
    text::{Span, Spans},
    widgets::{Block, BorderType, Cell, Row, Table},
};

use tui::widgets::Widget;

use std::{
    borrow::Cow,
    collections::HashMap,
    io::Read,
    path::Path,
    sync::{
        atomic::{self, AtomicUsize},
        Arc,
    },
};

use crate::ui::{Prompt, PromptEvent};
use helix_core::{
    char_idx_at_visual_offset, fuzzy::MATCHER, movement::Direction,
    text_annotations::TextAnnotations, unicode::segmentation::UnicodeSegmentation, Position,
};
use helix_view::{
    editor::Action,
    graphics::{CursorKind, Margin, Modifier, Rect},
    input::{MouseButton, MouseEvent, MouseEventKind},
    theme::Style,
    view::ViewPosition,
    Document, DocumentId, Editor,
};

use self::handlers::{DynamicQueryChange, DynamicQueryHandler, PreviewHighlightHandler};

pub const ID: &str = "picker";

pub const MIN_AREA_WIDTH_FOR_PREVIEW: u16 = 72;
/// Biggest file size to preview in bytes
pub const MAX_FILE_SIZE_FOR_PREVIEW: u64 = 10 * 1024 * 1024;

#[derive(PartialEq, Eq, Hash)]
pub enum PathOrId<'a> {
    Id(DocumentId),
    Path(&'a Path),
}

impl<'a> From<&'a Path> for PathOrId<'a> {
    fn from(path: &'a Path) -> Self {
        Self::Path(path)
    }
}

impl From<DocumentId> for PathOrId<'_> {
    fn from(v: DocumentId) -> Self {
        Self::Id(v)
    }
}

type FileCallback<T> = Box<dyn for<'a> Fn(&'a Editor, &'a T) -> Option<FileLocation<'a>>>;

/// File path and range of lines (used to align and highlight lines)
pub type FileLocation<'a> = (PathOrId<'a>, Option<(usize, usize)>);

pub enum CachedPreview {
    Document(Box<Document>),
    Directory(Vec<(String, bool)>),
    Binary,
    LargeFile,
    NotFound,
}

// We don't store this enum in the cache so as to avoid lifetime constraints
// from borrowing a document already opened in the editor.
pub enum Preview<'picker, 'editor> {
    Cached(&'picker CachedPreview),
    EditorDocument(&'editor Document),
}

impl Preview<'_, '_> {
    fn document(&self) -> Option<&Document> {
        match self {
            Preview::EditorDocument(doc) => Some(doc),
            Preview::Cached(CachedPreview::Document(doc)) => Some(doc),
            _ => None,
        }
    }

    fn dir_content(&self) -> Option<&Vec<(String, bool)>> {
        match self {
            Preview::Cached(CachedPreview::Directory(dir_content)) => Some(dir_content),
            _ => None,
        }
    }

    /// Alternate text to show for the preview.
    fn placeholder(&self) -> &str {
        match *self {
            Self::EditorDocument(_) => "<Invalid file location>",
            Self::Cached(preview) => match preview {
                CachedPreview::Document(_) => "<Invalid file location>",
                CachedPreview::Directory(_) => "<Invalid directory location>",
                CachedPreview::Binary => "<Binary file>",
                CachedPreview::LargeFile => "<File too large to preview>",
                CachedPreview::NotFound => "<File not found>",
            },
        }
    }
}

fn inject_nucleo_item<T, D>(
    injector: &nucleo::Injector<T>,
    columns: &[Column<T, D>],
    item: T,
    editor_data: &D,
) {
    injector.push(item, |item, dst| {
        for (column, text) in columns.iter().filter(|column| column.filter).zip(dst) {
            *text = column.format_text(item, editor_data).into()
        }
    });
}

pub struct Injector<T, D> {
    dst: nucleo::Injector<T>,
    columns: Arc<[Column<T, D>]>,
    editor_data: Arc<D>,
    version: usize,
    picker_version: Arc<AtomicUsize>,
    /// A marker that requests a redraw when the injector drops.
    /// This marker causes the "running" indicator to disappear when a background job
    /// providing items is finished and drops. This could be wrapped in an [Arc] to ensure
    /// that the redraw is only requested when all Injectors drop for a Picker (which removes
    /// the "running" indicator) but the redraw handle is debounced so this is unnecessary.
    _redraw: helix_event::RequestRedrawOnDrop,
}

impl<I, D> Clone for Injector<I, D> {
    fn clone(&self) -> Self {
        Injector {
            dst: self.dst.clone(),
            columns: self.columns.clone(),
            editor_data: self.editor_data.clone(),
            version: self.version,
            picker_version: self.picker_version.clone(),
            _redraw: helix_event::RequestRedrawOnDrop,
        }
    }
}

#[derive(Error, Debug)]
#[error("picker has been shut down")]
pub struct InjectorShutdown;

impl<T, D> Injector<T, D> {
    pub fn push(&self, item: T) -> Result<(), InjectorShutdown> {
        if self.version != self.picker_version.load(atomic::Ordering::Relaxed) {
            return Err(InjectorShutdown);
        }

        inject_nucleo_item(&self.dst, &self.columns, item, &self.editor_data);
        Ok(())
    }
}

type ColumnFormatFn<T, D> = for<'a> fn(&'a T, &'a D) -> Cell<'a>;

pub struct Column<T, D> {
    name: Arc<str>,
    format: ColumnFormatFn<T, D>,
    /// Whether the column should be passed to nucleo for matching and filtering.
    /// `DynamicPicker` uses this so that the dynamic column (for example regex in
    /// global search) is not used for filtering twice.
    filter: bool,
    hidden: bool,
}

impl<T, D> Column<T, D> {
    pub fn new(name: impl Into<Arc<str>>, format: ColumnFormatFn<T, D>) -> Self {
        Self {
            name: name.into(),
            format,
            filter: true,
            hidden: false,
        }
    }

    /// A column which does not display any contents
    pub fn hidden(name: impl Into<Arc<str>>) -> Self {
        let format = |_: &T, _: &D| unreachable!();

        Self {
            name: name.into(),
            format,
            filter: false,
            hidden: true,
        }
    }

    pub fn without_filtering(mut self) -> Self {
        self.filter = false;
        self
    }

    fn format<'a>(&self, item: &'a T, data: &'a D) -> Cell<'a> {
        (self.format)(item, data)
    }

    fn format_text<'a>(&self, item: &'a T, data: &'a D) -> Cow<'a, str> {
        let text: String = self.format(item, data).content.into();
        text.into()
    }
}

/// Returns a new list of options to replace the contents of the picker
/// when called with the current picker query,
type DynQueryCallback<T, D> =
    fn(&str, &mut Editor, Arc<D>, &Injector<T, D>) -> BoxFuture<'static, anyhow::Result<()>>;

pub struct Picker<T: 'static + Send + Sync, D: 'static> {
    columns: Arc<[Column<T, D>]>,
    primary_column: usize,
    editor_data: Arc<D>,
    version: Arc<AtomicUsize>,
    matcher: Nucleo<T>,

    /// Current height of the completions box
    completion_height: u16,

    cursor: u32,
    prompt: Prompt,
    query: PickerQuery,

    /// Whether to show the preview panel (default true)
    show_preview: bool,
    /// Constraints for tabular formatting
    widths: Vec<Constraint>,

    callback_fn: PickerCallback<T>,
    default_action: Action,

    pub truncate_start: bool,
    /// Caches paths to documents
    preview_cache: HashMap<Arc<Path>, CachedPreview>,
    read_buffer: Vec<u8>,
    /// Given an item in the picker, return the file path and line number to display.
    file_fn: Option<FileCallback<T>>,
    /// An event handler for syntax highlighting the currently previewed file.
    preview_highlight_handler: Sender<Arc<Path>>,
    dynamic_query_handler: Option<Sender<DynamicQueryChange>>,
    /// Current render area, stored for mouse coordinate mapping
    current_area: Option<Rect>,
    /// Scroll offset for the preview panel
    preview_offset: Option<ViewPosition>,
    /// Pending scroll delta to apply in render_preview (visual lines)
    pending_preview_scroll: isize,
}

impl<T: 'static + Send + Sync, D: 'static + Send + Sync> Picker<T, D> {
    pub fn stream(
        columns: impl IntoIterator<Item = Column<T, D>>,
        editor_data: D,
    ) -> (Nucleo<T>, Injector<T, D>) {
        let columns: Arc<[_]> = columns.into_iter().collect();
        let matcher_columns = columns.iter().filter(|col| col.filter).count() as u32;
        assert!(matcher_columns > 0);
        let matcher = Nucleo::new(
            Config::DEFAULT,
            Arc::new(helix_event::request_redraw),
            None,
            matcher_columns,
        );
        let streamer = Injector {
            dst: matcher.injector(),
            columns,
            editor_data: Arc::new(editor_data),
            version: 0,
            picker_version: Arc::new(AtomicUsize::new(0)),
            _redraw: helix_event::RequestRedrawOnDrop,
        };
        (matcher, streamer)
    }

    pub fn new<C, O, F>(
        columns: C,
        primary_column: usize,
        options: O,
        editor_data: D,
        callback_fn: F,
    ) -> Self
    where
        C: IntoIterator<Item = Column<T, D>>,
        O: IntoIterator<Item = T>,
        F: Fn(&mut Context, &T, Action) + 'static,
    {
        let columns: Arc<[_]> = columns.into_iter().collect();
        let matcher_columns = columns
            .iter()
            .filter(|col: &&Column<T, D>| col.filter)
            .count() as u32;
        assert!(matcher_columns > 0);
        let matcher = Nucleo::new(
            Config::DEFAULT,
            Arc::new(helix_event::request_redraw),
            None,
            matcher_columns,
        );
        let injector = matcher.injector();
        for item in options {
            inject_nucleo_item(&injector, &columns, item, &editor_data);
        }
        Self::with(
            matcher,
            columns,
            primary_column,
            Arc::new(editor_data),
            Arc::new(AtomicUsize::new(0)),
            callback_fn,
        )
    }

    pub fn with_stream(
        matcher: Nucleo<T>,
        primary_column: usize,
        injector: Injector<T, D>,
        callback_fn: impl Fn(&mut Context, &T, Action) + 'static,
    ) -> Self {
        Self::with(
            matcher,
            injector.columns,
            primary_column,
            injector.editor_data,
            injector.picker_version,
            callback_fn,
        )
    }

    fn with(
        matcher: Nucleo<T>,
        columns: Arc<[Column<T, D>]>,
        default_column: usize,
        editor_data: Arc<D>,
        version: Arc<AtomicUsize>,
        callback_fn: impl Fn(&mut Context, &T, Action) + 'static,
    ) -> Self {
        assert!(!columns.is_empty());

        let prompt = Prompt::new(
            "".into(),
            None,
            ui::completers::none,
            |_editor: &mut Context, _pattern: &str, _event: PromptEvent| {},
        );

        let widths = columns
            .iter()
            .map(|column| Constraint::Length(column.name.chars().count() as u16))
            .collect();

        let query = PickerQuery::new(columns.iter().map(|col| &col.name).cloned(), default_column);

        Self {
            columns,
            primary_column: default_column,
            matcher,
            editor_data,
            version,
            cursor: 0,
            prompt,
            query,
            truncate_start: true,
            show_preview: true,
            callback_fn: Box::new(callback_fn),
            default_action: Action::Replace,
            completion_height: 0,
            widths,
            preview_cache: HashMap::new(),
            read_buffer: Vec::with_capacity(1024),
            file_fn: None,
            preview_highlight_handler: PreviewHighlightHandler::<T, D>::default().spawn(),
            dynamic_query_handler: None,
            current_area: None,
            preview_offset: None,
            pending_preview_scroll: 0,
        }
    }

    pub fn injector(&self) -> Injector<T, D> {
        Injector {
            dst: self.matcher.injector(),
            columns: self.columns.clone(),
            editor_data: self.editor_data.clone(),
            version: self.version.load(atomic::Ordering::Relaxed),
            picker_version: self.version.clone(),
            _redraw: helix_event::RequestRedrawOnDrop,
        }
    }

    pub fn truncate_start(mut self, truncate_start: bool) -> Self {
        self.truncate_start = truncate_start;
        self
    }

    pub fn with_preview(
        mut self,
        preview_fn: impl for<'a> Fn(&'a Editor, &'a T) -> Option<FileLocation<'a>> + 'static,
    ) -> Self {
        self.file_fn = Some(Box::new(preview_fn));
        // assumption: if we have a preview we are matching paths... If this is ever
        // not true this could be a separate builder function
        self.matcher.update_config(Config::DEFAULT.match_paths());
        self
    }

    pub fn with_history_register(mut self, history_register: Option<char>) -> Self {
        self.prompt.with_history_register(history_register);
        self
    }

    pub fn with_initial_cursor(mut self, cursor: u32) -> Self {
        self.cursor = cursor;
        self
    }

    pub fn with_dynamic_query(
        mut self,
        callback: DynQueryCallback<T, D>,
        debounce_ms: Option<u64>,
    ) -> Self {
        let handler = DynamicQueryHandler::new(callback, debounce_ms).spawn();
        let event = DynamicQueryChange {
            query: self.primary_query(),
            // Treat the initial query as a paste.
            is_paste: true,
        };
        helix_event::send_blocking(&handler, event);
        self.dynamic_query_handler = Some(handler);
        self
    }

    pub fn with_default_action(mut self, action: Action) -> Self {
        self.default_action = action;
        self
    }

    /// Move the cursor by a number of lines, either down (`Forward`) or up (`Backward`)
    pub fn move_by(&mut self, amount: u32, direction: Direction) {
        let len = self.matcher.snapshot().matched_item_count();

        if len == 0 {
            // No results, can't move.
            return;
        }

        match direction {
            Direction::Forward => {
                self.cursor = self.cursor.saturating_add(amount) % len;
                // Reset preview offset when cursor changes (new selection)
                self.preview_offset = None;
                self.pending_preview_scroll = 0;
            }
            Direction::Backward => {
                self.cursor = self.cursor.saturating_add(len).saturating_sub(amount) % len;
                // Reset preview offset when cursor changes (new selection)
                self.preview_offset = None;
                self.pending_preview_scroll = 0;
            }
        }
    }

    /// Move the cursor down by exactly one page. After the last page comes the first page.
    pub fn page_up(&mut self) {
        self.move_by(self.completion_height as u32, Direction::Backward);
    }

    /// Move the cursor up by exactly one page. After the first page comes the last page.
    pub fn page_down(&mut self) {
        self.move_by(self.completion_height as u32, Direction::Forward);
    }

    /// Move the cursor to the first entry
    pub fn to_start(&mut self) {
        self.cursor = 0;
        // Reset preview offset when cursor changes (new selection)
        self.preview_offset = None;
        self.pending_preview_scroll = 0;
    }

    /// Move the cursor to the last entry
    pub fn to_end(&mut self) {
        self.cursor = self
            .matcher
            .snapshot()
            .matched_item_count()
            .saturating_sub(1);
        // Reset preview offset when cursor changes (new selection)
        self.preview_offset = None;
        self.pending_preview_scroll = 0;
    }

    pub fn selection(&self) -> Option<&T> {
        self.matcher
            .snapshot()
            .get_matched_item(self.cursor)
            .map(|item| item.data)
    }

    fn primary_query(&self) -> Arc<str> {
        self.query
            .get(&self.columns[self.primary_column].name)
            .cloned()
            .unwrap_or_else(|| "".into())
    }

    fn header_height(&self) -> u16 {
        if self.columns.len() > 1 {
            1
        } else {
            0
        }
    }

    /// Convert mouse Y coordinate to row index in the picker list
    /// Returns None if the coordinate is not within the table area
    fn mouse_y_to_row_index(&self, mouse_y: u16, picker_area: Rect) -> Option<usize> {
        // Convert absolute screen Y to relative Y within picker
        let relative_y = mouse_y.saturating_sub(picker_area.top());

        // Skip border (1), prompt (1), separator (1) = 3 rows
        if relative_y < 3 {
            return None;
        }

        // Calculate row in table (after header)
        let table_relative_y = relative_y - 3;
        let header_height = self.header_height();

        if table_relative_y < header_height {
            return None; // Clicked header
        }

        // Row index in visible items
        let row_in_view = (table_relative_y - header_height) as usize;

        // Account for scroll offset
        let rows_visible = self.completion_height as u32;
        let scroll_offset = self.cursor - (self.cursor % std::cmp::max(1, rows_visible));

        // Calculate actual index
        let actual_index = scroll_offset as usize + row_in_view;

        // Check bounds
        let total_items = self.matcher.snapshot().matched_item_count() as usize;
        if actual_index >= total_items {
            return None;
        }

        Some(actual_index)
    }

    /// Handle mouse events for scrolling and selection
    fn handle_mouse_event(
        &mut self,
        event: &MouseEvent,
        scroll_lines: usize,
        cx: &mut Context,
    ) -> EventResult {
        let Some(area) = self.current_area else {
            return EventResult::Ignored(None);
        };

        let MouseEvent {
            kind,
            column: x,
            row: y,
            ..
        } = *event;

        // Calculate picker area (considering preview panel)
        let render_preview =
            self.show_preview && self.file_fn.is_some() && area.width > MIN_AREA_WIDTH_FOR_PREVIEW;
        let picker_width = if render_preview {
            area.width / 2
        } else {
            area.width
        };
        let picker_area = area.with_width(picker_width);

        // Check if mouse is within picker bounds (similar to Popup)
        let mouse_is_within_picker = x >= picker_area.left()
            && x < picker_area.right()
            && y >= picker_area.top()
            && y < picker_area.bottom();

        // Check if mouse is within preview bounds
        let mouse_is_within_preview = if render_preview {
            let preview_area = area.clip_left(picker_width);
            x >= preview_area.left()
                && x < preview_area.right()
                && y >= preview_area.top()
                && y < preview_area.bottom()
        } else {
            false
        };

        // If mouse is not in either area, ignore the event
        if !mouse_is_within_picker && !mouse_is_within_preview {
            return EventResult::Ignored(None);
        }

        match kind {
            MouseEventKind::ScrollDown => {
                if mouse_is_within_picker {
                    // Scroll picker list one item at a time
                    self.move_by(1, Direction::Forward);
                    EventResult::Consumed(None)
                } else if mouse_is_within_preview {
                    // Scroll preview panel using config.scroll_lines
                    self.pending_preview_scroll = self
                        .pending_preview_scroll
                        .saturating_add(scroll_lines as isize);
                    EventResult::Consumed(None)
                } else {
                    EventResult::Ignored(None)
                }
            }
            MouseEventKind::ScrollUp => {
                if mouse_is_within_picker {
                    // Scroll picker list one item at a time
                    self.move_by(1, Direction::Backward);
                    EventResult::Consumed(None)
                } else if mouse_is_within_preview {
                    // Scroll preview panel using config.scroll_lines
                    self.pending_preview_scroll = self
                        .pending_preview_scroll
                        .saturating_sub(scroll_lines as isize);
                    EventResult::Consumed(None)
                } else {
                    EventResult::Ignored(None)
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if mouse_is_within_picker {
                    // Single click in picker area: select item
                    if let Some(row_index) = self.mouse_y_to_row_index(y, picker_area) {
                        self.cursor = row_index as u32;
                        // Reset preview offset when cursor changes (new selection)
                        self.preview_offset = None;
                        self.pending_preview_scroll = 0;
                        EventResult::Consumed(None)
                    } else {
                        EventResult::Ignored(None)
                    }
                } else if mouse_is_within_preview {
                    // Click in preview area: open the file
                    if let Some(option) = self.selection() {
                        // Open the file
                        (self.callback_fn)(cx, option, self.default_action);
                        // Return a callback to close the picker
                        return EventResult::Consumed(Some(Box::new(
                            |compositor: &mut Compositor, _ctx| {
                                compositor.pop();
                            },
                        )));
                    }
                    EventResult::Consumed(None)
                } else {
                    EventResult::Ignored(None)
                }
            }
            _ => EventResult::Ignored(None),
        }
    }

    pub fn toggle_preview(&mut self) {
        self.show_preview = !self.show_preview;
        // Reset preview offset when preview is toggled
        self.preview_offset = None;
        self.pending_preview_scroll = 0;
    }

    fn prompt_handle_event(&mut self, event: &Event, cx: &mut Context) -> EventResult {
        if let EventResult::Consumed(_) = self.prompt.handle_event(event, cx) {
            self.handle_prompt_change(matches!(event, Event::Paste(_)));
        }
        EventResult::Consumed(None)
    }

    fn handle_prompt_change(&mut self, is_paste: bool) {
        // TODO: better track how the pattern has changed
        let line = self.prompt.line();
        let old_query = self.query.parse(line);
        if self.query == old_query {
            return;
        }
        // If the query has meaningfully changed, reset the cursor to the top of the results.
        self.cursor = 0;
        // Have nucleo reparse each changed column.
        for (i, column) in self
            .columns
            .iter()
            .filter(|column| column.filter)
            .enumerate()
        {
            let pattern = self
                .query
                .get(&column.name)
                .map(|f| &**f)
                .unwrap_or_default();
            let old_pattern = old_query
                .get(&column.name)
                .map(|f| &**f)
                .unwrap_or_default();
            // Fastlane: most columns will remain unchanged after each edit.
            if pattern == old_pattern {
                continue;
            }
            let is_append = pattern.starts_with(old_pattern);
            self.matcher.pattern.reparse(
                i,
                pattern,
                CaseMatching::Smart,
                Normalization::Smart,
                is_append,
            );
        }
        // If this is a dynamic picker, notify the query hook that the primary
        // query might have been updated.
        if let Some(handler) = &self.dynamic_query_handler {
            let event = DynamicQueryChange {
                query: self.primary_query(),
                is_paste,
            };
            helix_event::send_blocking(handler, event);
        }
    }

    /// Get (cached) preview for the currently selected item. If a document corresponding
    /// to the path is already open in the editor, it is used instead.
    fn get_preview<'picker, 'editor>(
        &'picker mut self,
        editor: &'editor Editor,
    ) -> Option<(Preview<'picker, 'editor>, Option<(usize, usize)>)> {
        let current = self.selection()?;
        let (path_or_id, range) = (self.file_fn.as_ref()?)(editor, current)?;

        match path_or_id {
            PathOrId::Path(path) => {
                if let Some(doc) = editor.document_by_path(path) {
                    return Some((Preview::EditorDocument(doc), range));
                }

                if self.preview_cache.contains_key(path) {
                    // NOTE: we use `HashMap::get_key_value` here instead of indexing so we can
                    // retrieve the `Arc<Path>` key. The `path` in scope here is a `&Path` and
                    // we can cheaply clone the key for the preview highlight handler.
                    let (path, preview) = self.preview_cache.get_key_value(path).unwrap();
                    if matches!(preview, CachedPreview::Document(doc) if doc.syntax().is_none()) {
                        helix_event::send_blocking(&self.preview_highlight_handler, path.clone());
                    }
                    return Some((Preview::Cached(preview), range));
                }

                let path: Arc<Path> = path.into();
                let preview = std::fs::metadata(&path)
                    .and_then(|metadata| {
                        if metadata.is_dir() {
                            let files = super::directory_content(&path, editor)?;
                            let file_names: Vec<_> = files
                                .iter()
                                .filter_map(|(file_path, is_dir)| {
                                    let name = file_path
                                        .strip_prefix(&path)
                                        .map(|p| Some(p.as_os_str()))
                                        .unwrap_or_else(|_| file_path.file_name())?
                                        .to_string_lossy();
                                    if *is_dir {
                                        Some((format!("{}/", name), true))
                                    } else {
                                        Some((name.into_owned(), false))
                                    }
                                })
                                .collect();
                            Ok(CachedPreview::Directory(file_names))
                        } else if metadata.is_file() {
                            if metadata.len() > MAX_FILE_SIZE_FOR_PREVIEW {
                                return Ok(CachedPreview::LargeFile);
                            }
                            let content_type = std::fs::File::open(&path).and_then(|file| {
                                // Read up to 1kb to detect the content type
                                let n = file.take(1024).read_to_end(&mut self.read_buffer)?;
                                let content_type =
                                    content_inspector::inspect(&self.read_buffer[..n]);
                                self.read_buffer.clear();
                                Ok(content_type)
                            })?;
                            if content_type.is_binary() {
                                return Ok(CachedPreview::Binary);
                            }
                            let mut doc = Document::open(
                                &path,
                                None,
                                false,
                                editor.config.clone(),
                                editor.syn_loader.clone(),
                            )
                            .or(Err(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                "Cannot open document",
                            )))?;
                            let loader = editor.syn_loader.load();
                            if let Some(language_config) = doc.detect_language_config(&loader) {
                                doc.language = Some(language_config);
                                // Asynchronously highlight the new document
                                helix_event::send_blocking(
                                    &self.preview_highlight_handler,
                                    path.clone(),
                                );
                            }
                            Ok(CachedPreview::Document(Box::new(doc)))
                        } else {
                            Err(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                "Neither a dir, nor a file",
                            ))
                        }
                    })
                    .unwrap_or(CachedPreview::NotFound);
                self.preview_cache.insert(path.clone(), preview);
                Some((Preview::Cached(&self.preview_cache[&path]), range))
            }
            PathOrId::Id(id) => {
                let doc = editor.documents.get(&id).unwrap();
                Some((Preview::EditorDocument(doc), range))
            }
        }
    }

    fn render_picker(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let status = self.matcher.tick(10);
        let snapshot = self.matcher.snapshot();
        if status.changed {
            self.cursor = self
                .cursor
                .min(snapshot.matched_item_count().saturating_sub(1))
        }

        let text_style = cx.editor.theme.get("ui.text");
        let selected = cx.editor.theme.get("ui.text.focus");
        let highlight_style = cx.editor.theme.get("special").add_modifier(Modifier::BOLD);

        // -- Render the frame:
        // clear area
        let background = cx.editor.theme.get("ui.background");
        surface.clear_with(area, background);

        const BLOCK: Block<'_> = Block::bordered();

        // calculate the inner area inside the box
        let inner = BLOCK.inner(area);

        BLOCK.render(area, surface);

        // -- Render the input bar:

        let count = format!(
            "{}{}/{}",
            if status.running || self.matcher.active_injectors() > 0 {
                "(running) "
            } else {
                ""
            },
            snapshot.matched_item_count(),
            snapshot.item_count(),
        );

        let area = inner.clip_left(1).with_height(1);
        let line_area = area.clip_right(count.len() as u16 + 1);

        // render the prompt first since it will clear its background
        self.prompt.render(line_area, surface, cx);

        surface.set_stringn(
            (area.x + area.width).saturating_sub(count.len() as u16 + 1),
            area.y,
            &count,
            (count.len()).min(area.width as usize),
            text_style,
        );

        // -- Separator
        let sep_style = cx.editor.theme.get("ui.background.separator");
        let borders = BorderType::line_symbols(BorderType::Plain);
        for x in inner.left()..inner.right() {
            if let Some(cell) = surface.get_mut(x, inner.y + 1) {
                cell.set_symbol(borders.horizontal).set_style(sep_style);
            }
        }

        // -- Render the contents:
        // subtract area of prompt from top
        let inner = inner.clip_top(2);
        let rows = inner.height.saturating_sub(self.header_height()) as u32;
        let offset = self.cursor - (self.cursor % std::cmp::max(1, rows));
        let cursor = self.cursor.saturating_sub(offset);
        let end = offset
            .saturating_add(rows)
            .min(snapshot.matched_item_count());
        let mut indices = Vec::new();
        let mut matcher = MATCHER.lock();
        matcher.config = Config::DEFAULT;
        if self.file_fn.is_some() {
            matcher.config.set_match_paths()
        }

        let options = snapshot.matched_items(offset..end).map(|item| {
            let mut widths = self.widths.iter_mut();
            let mut matcher_index = 0;

            Row::new(self.columns.iter().map(|column| {
                if column.hidden {
                    return Cell::default();
                }

                let Some(Constraint::Length(max_width)) = widths.next() else {
                    unreachable!();
                };
                let mut cell = column.format(item.data, &self.editor_data);
                let width = if column.filter {
                    snapshot.pattern().column_pattern(matcher_index).indices(
                        item.matcher_columns[matcher_index].slice(..),
                        &mut matcher,
                        &mut indices,
                    );
                    indices.sort_unstable();
                    indices.dedup();
                    let mut indices = indices.drain(..);
                    let mut next_highlight_idx = indices.next().unwrap_or(u32::MAX);
                    let mut span_list = Vec::new();
                    let mut current_span = String::new();
                    let mut current_style = Style::default();
                    let mut grapheme_idx = 0u32;
                    let mut width = 0;

                    let spans: &[Span] =
                        cell.content.lines.first().map_or(&[], |it| it.0.as_slice());
                    for span in spans {
                        // this looks like a bug on first glance, we are iterating
                        // graphemes but treating them as char indices. The reason that
                        // this is correct is that nucleo will only ever consider the first char
                        // of a grapheme (and discard the rest of the grapheme) so the indices
                        // returned by nucleo are essentially grapheme indecies
                        for grapheme in span.content.graphemes(true) {
                            let style = if grapheme_idx == next_highlight_idx {
                                next_highlight_idx = indices.next().unwrap_or(u32::MAX);
                                span.style.patch(highlight_style)
                            } else {
                                span.style
                            };
                            if style != current_style {
                                if !current_span.is_empty() {
                                    span_list.push(Span::styled(current_span, current_style))
                                }
                                current_span = String::new();
                                current_style = style;
                            }
                            current_span.push_str(grapheme);
                            grapheme_idx += 1;
                        }
                        width += span.width();
                    }

                    span_list.push(Span::styled(current_span, current_style));
                    cell = Cell::from(Spans::from(span_list));
                    matcher_index += 1;
                    width
                } else {
                    cell.content
                        .lines
                        .first()
                        .map(|line| line.width())
                        .unwrap_or_default()
                };

                if width as u16 > *max_width {
                    *max_width = width as u16;
                }

                cell
            }))
        });

        let mut table = Table::new(options)
            .style(text_style)
            .highlight_style(selected)
            .highlight_symbol(" > ")
            .column_spacing(1)
            .widths(&self.widths);

        // -- Header
        if self.columns.len() > 1 {
            let active_column = self.query.active_column(self.prompt.position());
            let header_style = cx.editor.theme.get("ui.picker.header");
            let header_column_style = cx.editor.theme.get("ui.picker.header.column");

            table = table.header(
                Row::new(self.columns.iter().map(|column| {
                    if column.hidden {
                        Cell::default()
                    } else {
                        let style =
                            if active_column.is_some_and(|name| Arc::ptr_eq(name, &column.name)) {
                                cx.editor.theme.get("ui.picker.header.column.active")
                            } else {
                                header_column_style
                            };

                        Cell::from(Span::styled(Cow::from(&*column.name), style))
                    }
                }))
                .style(header_style),
            );
        }

        use tui::widgets::TableState;

        table.render_table(
            inner,
            surface,
            &mut TableState {
                offset: 0,
                selected: Some(cursor as usize),
            },
            self.truncate_start,
        );
    }

    fn render_preview(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        // -- Render the frame:
        // clear area
        let background = cx.editor.theme.get("ui.background");
        let text = cx.editor.theme.get("ui.text");
        let directory = cx.editor.theme.get("ui.text.directory");
        surface.clear_with(area, background);

        const BLOCK: Block<'_> = Block::bordered();

        // calculate the inner area inside the box
        let inner = BLOCK.inner(area);
        // 1 column gap on either side
        let margin = Margin::horizontal(1);
        let inner = inner.inner(margin);
        BLOCK.render(area, surface);

        // Get offset and pending scroll delta before borrowing self
        let has_offset = self.preview_offset.is_some();
        let mut offset = self.preview_offset.unwrap_or_default();
        let mut needs_store = false;
        let scroll_delta = self.pending_preview_scroll;

        if let Some((preview, range)) = self.get_preview(cx.editor) {
            let doc = match preview.document() {
                Some(doc)
                    if range.is_none_or(|(start, end)| {
                        start <= end && end <= doc.text().len_lines()
                    }) =>
                {
                    doc
                }
                _ => {
                    if let Some(dir_content) = preview.dir_content() {
                        for (i, (path, is_dir)) in
                            dir_content.iter().take(inner.height as usize).enumerate()
                        {
                            let style = if *is_dir { directory } else { text };
                            surface.set_stringn(
                                inner.x,
                                inner.y + i as u16,
                                path,
                                inner.width as usize,
                                style,
                            );
                        }
                        return;
                    }

                    let alt_text = preview.placeholder();
                    let x = inner.x + inner.width.saturating_sub(alt_text.len() as u16) / 2;
                    let y = inner.y + inner.height / 2;
                    surface.set_stringn(x, y, alt_text, inner.width as usize, text);
                    return;
                }
            };

            // If we don't have an offset, calculate it based on range
            if !has_offset {
                if let Some((start_line, end_line)) = range {
                    let height = end_line - start_line;
                    let text = doc.text().slice(..);
                    let start = text.line_to_char(start_line);
                    let middle = text.line_to_char(start_line + height / 2);
                    if height < inner.height as usize {
                        let text_fmt = doc.text_format(inner.width, None);
                        let annotations = TextAnnotations::default();
                        (offset.anchor, offset.vertical_offset) = char_idx_at_visual_offset(
                            text,
                            middle,
                            // align to middle
                            -(inner.height as isize / 2),
                            0,
                            &text_fmt,
                            &annotations,
                        );
                        if start < offset.anchor {
                            offset.anchor = start;
                            offset.vertical_offset = 0;
                        }
                    } else {
                        offset.anchor = start;
                    }
                }
            }

            // Apply pending scroll delta
            if scroll_delta != 0 {
                let text = doc.text().slice(..);
                let text_fmt = doc.text_format(inner.width, None);
                let annotations = TextAnnotations::default();

                (offset.anchor, offset.vertical_offset) = char_idx_at_visual_offset(
                    text,
                    offset.anchor,
                    offset.vertical_offset as isize + scroll_delta,
                    0,
                    &text_fmt,
                    &annotations,
                );

                // Clamp scroll to keep at least scrolloff lines visible
                let scrolloff = cx.editor.config().scrolloff;
                let total_lines = text.len_lines();
                let anchor_line = text.char_to_line(offset.anchor);
                let max_line = total_lines.saturating_sub(scrolloff.max(1) + 1);
                if anchor_line > max_line {
                    offset.anchor = text.line_to_char(max_line);
                    offset.vertical_offset = 0;
                }

                needs_store = true;
            }
            let loader = cx.editor.syn_loader.load();
            let config = cx.editor.config();

            // Ensure vertical offset is not negative
            offset.vertical_offset = offset.vertical_offset.max(0);

            let syntax_highlighter =
                EditorView::doc_syntax_highlighter(doc, offset.anchor, area.height, &loader);
            let mut overlay_highlights = Vec::new();
            if doc
                .language_config()
                .and_then(|config| config.rainbow_brackets)
                .unwrap_or(config.rainbow_brackets)
            {
                if let Some(overlay) = EditorView::doc_rainbow_highlights(
                    doc,
                    offset.anchor,
                    area.height,
                    &cx.editor.theme,
                    &loader,
                ) {
                    overlay_highlights.push(overlay);
                }
            }

            EditorView::doc_diagnostics_highlights_into(
                doc,
                &cx.editor.theme,
                &mut overlay_highlights,
            );

            let mut decorations = DecorationManager::default();

            if let Some((start, end)) = range {
                let style = cx
                    .editor
                    .theme
                    .try_get("ui.highlight")
                    .unwrap_or_else(|| cx.editor.theme.get("ui.selection"));
                let draw_highlight = move |renderer: &mut TextRenderer, pos: LinePos| {
                    if (start..=end).contains(&pos.doc_line) {
                        let area = Rect::new(
                            renderer.viewport.x,
                            pos.visual_line,
                            renderer.viewport.width,
                            1,
                        );
                        renderer.set_style(area, style)
                    }
                };
                decorations.add_decoration(draw_highlight);
            }

            render_document(
                surface,
                inner,
                doc,
                offset,
                // TODO: compute text annotations asynchronously here (like inlay hints)
                &TextAnnotations::default(),
                syntax_highlighter,
                overlay_highlights,
                &cx.editor.theme,
                decorations,
            );
        }

        // Store the updated offset if we calculated one
        if needs_store {
            self.preview_offset = Some(offset);
            self.pending_preview_scroll = 0;
        }
    }
}

impl<I: 'static + Send + Sync, D: 'static + Send + Sync> Component for Picker<I, D> {
    fn render(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        // Store current area for mouse coordinate mapping
        self.current_area = Some(area);

        // +---------+ +---------+
        // |prompt   | |preview  |
        // +---------+ |         |
        // |picker   | |         |
        // |         | |         |
        // +---------+ +---------+

        let render_preview =
            self.show_preview && self.file_fn.is_some() && area.width > MIN_AREA_WIDTH_FOR_PREVIEW;

        let picker_width = if render_preview {
            area.width / 2
        } else {
            area.width
        };

        let picker_area = area.with_width(picker_width);
        self.render_picker(picker_area, surface, cx);

        if render_preview {
            let preview_area = area.clip_left(picker_width);
            self.render_preview(preview_area, surface, cx);
        }
    }

    fn handle_event(&mut self, event: &Event, ctx: &mut Context) -> EventResult {
        // TODO: keybinds for scrolling preview

        match event {
            Event::Key(event) => {
                let key_event = *event;

                let close_fn = |picker: &mut Self| {
                    // if the picker is very large don't store it as last_picker to avoid
                    // excessive memory consumption
                    let callback: compositor::Callback =
                        if picker.matcher.snapshot().item_count() > 1_000_000 {
                            Box::new(|compositor: &mut Compositor, _ctx| {
                                // remove the layer
                                compositor.pop();
                            })
                        } else {
                            // stop streaming in new items in the background, really we should
                            // be restarting the stream somehow once the picker gets
                            // reopened instead (like for an FS crawl) that would also remove the
                            // need for the special case above but that is pretty tricky
                            picker.version.fetch_add(1, atomic::Ordering::Relaxed);
                            Box::new(|compositor: &mut Compositor, _ctx| {
                                // remove the layer
                                compositor.last_picker = compositor.pop();
                            })
                        };
                    EventResult::Consumed(Some(callback))
                };

                match key_event {
                    shift!(Tab) | key!(Up) | ctrl!('p') => {
                        self.move_by(1, Direction::Backward);
                    }
                    key!(Tab) | key!(Down) | ctrl!('n') => {
                        self.move_by(1, Direction::Forward);
                    }
                    key!(PageDown) | ctrl!('d') => {
                        self.page_down();
                    }
                    key!(PageUp) | ctrl!('u') => {
                        self.page_up();
                    }
                    key!(Home) => {
                        self.to_start();
                    }
                    key!(End) => {
                        self.to_end();
                    }
                    key!(Esc) | ctrl!('c') => return close_fn(self),
                    alt!(Enter) => {
                        if let Some(option) = self.selection() {
                            (self.callback_fn)(ctx, option, self.default_action);
                        }
                    }
                    key!(Enter) => {
                        // If the prompt has a history completion and is empty, use enter to accept
                        // that completion
                        if let Some(completion) = self
                            .prompt
                            .first_history_completion(ctx.editor)
                            .filter(|_| self.prompt.line().is_empty())
                        {
                            // The percent character is used by the query language and needs to be
                            // escaped with a backslash.
                            let completion = if completion.contains('%') {
                                completion.replace('%', "\\%")
                            } else {
                                completion.into_owned()
                            };
                            self.prompt.set_line(completion, ctx.editor);

                            // Inserting from the history register is a paste.
                            self.handle_prompt_change(true);
                        } else {
                            if let Some(option) = self.selection() {
                                (self.callback_fn)(ctx, option, self.default_action);
                            }
                            if let Some(history_register) = self.prompt.history_register() {
                                if let Err(err) = ctx
                                    .editor
                                    .registers
                                    .push(history_register, self.primary_query().to_string())
                                {
                                    ctx.editor.set_error(err.to_string());
                                }
                            }
                            return close_fn(self);
                        }
                    }
                    ctrl!('s') => {
                        if let Some(option) = self.selection() {
                            (self.callback_fn)(ctx, option, Action::HorizontalSplit);
                        }
                        return close_fn(self);
                    }
                    ctrl!('v') => {
                        if let Some(option) = self.selection() {
                            (self.callback_fn)(ctx, option, Action::VerticalSplit);
                        }
                        return close_fn(self);
                    }
                    ctrl!('t') => {
                        self.toggle_preview();
                    }
                    _ => {
                        // Pass the original event (Event::Key) to prompt_handle_event
                        self.prompt_handle_event(&Event::Key(key_event), ctx);
                    }
                }

                EventResult::Consumed(None)
            }
            Event::Paste(..) => return self.prompt_handle_event(event, ctx),
            Event::Resize(..) => return EventResult::Consumed(None),
            Event::Mouse(event) => {
                let scroll_lines = ctx.editor.config().scroll_lines.unsigned_abs();
                return self.handle_mouse_event(event, scroll_lines, ctx);
            }
            _ => return EventResult::Ignored(None),
        }
    }

    fn cursor(&self, area: Rect, editor: &Editor) -> (Option<Position>, CursorKind) {
        let block = Block::bordered();
        // calculate the inner area inside the box
        let inner = block.inner(area);

        // prompt area
        let render_preview =
            self.show_preview && self.file_fn.is_some() && area.width > MIN_AREA_WIDTH_FOR_PREVIEW;

        let picker_width = if render_preview {
            area.width / 2
        } else {
            area.width
        };
        let area = inner.clip_left(1).with_height(1).with_width(picker_width);

        self.prompt.cursor(area, editor)
    }

    fn required_size(&mut self, (width, height): (u16, u16)) -> Option<(u16, u16)> {
        self.completion_height = height.saturating_sub(4 + self.header_height());
        Some((width, height))
    }

    fn id(&self) -> Option<&'static str> {
        Some(ID)
    }
}
impl<T: 'static + Send + Sync, D> Drop for Picker<T, D> {
    fn drop(&mut self) {
        // ensure we cancel any ongoing background threads streaming into the picker
        self.version.fetch_add(1, atomic::Ordering::Relaxed);
    }
}

type PickerCallback<T> = Box<dyn Fn(&mut Context, &T, Action)>;
