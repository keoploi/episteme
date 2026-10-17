/*
 * Episteme Reader - A native Android document reader.
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * mail: epistemereader@gmail.com
 */
package com.aryan.reader

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.aryan.reader.pdf.OcrLanguage
import com.aryan.reader.pdf.ocr.OcrBlock
import com.aryan.reader.pdf.ocr.OcrElement
import com.aryan.reader.pdf.ocr.OcrLine
import com.aryan.reader.pdf.ocr.OcrResult
import com.aryan.reader.pdf.ocr.OcrSymbol
import com.googlecode.tesseract.android.ResultIterator
import com.googlecode.tesseract.android.TessBaseAPI
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Offline OCR for the OSS build, backed by Tesseract (tesseract4android-openmp).
 *
 * Implements the same contract as the Play flavor's ML Kit engine: `init`/
 * `extractTextFromBitmap` returning an [OcrResult] whose boxes live in the
 * coordinate space of the bitmap that was passed in (ML Kit parity), so search
 * highlights and text selection land in the same places without coordinate
 * translation. Persian text comes back in logical order because the
 * ResultIterator visits RTL words in reading order.
 *
 * Threading: `TessBaseAPI` is not thread-safe, so a small pool of independent
 * instances is kept instead of one shared engine: background page indexing then
 * OCRs two pages in parallel (LSTM inference is single-threaded per instance),
 * and each slot serializes its own accesses behind its own mutex.
 * The engines live for the app's lifetime like the ML Kit recognizer; only a
 * language switch ([init]) or [release] recycles them.
 */
object OcrEngine {
    private const val TAG = "OcrEngine"

    private const val MAX_OCR_WIDTH = 1600
    private const val TESSDATA_DIR = "tessdata"
    private const val MIN_TRAINEDDATA_BYTES = 1000

    // PERSIAN_FAST decodes with the Persian model only: Tesseract merges every
    // listed language into one engine, so listing `eng` here roughly doubles
    // per-page recognition work. Language choice notes in `OcrLanguage` match.
    private const val LANGS_PERSIAN_PLUS_ENGLISH = "fas+eng"
    private const val LANGS_PERSIAN_ONLY = "fas"
    private const val LANGS_FALLBACK = "eng"

    @Volatile
    private var appContext: Context? = null

    @Volatile
    private var requestedLangs: String = LANGS_FALLBACK

    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** One Tesseract instance per slot; each slot guards itself with [gate]. */
    private class EngineSlot {
        val gate = Mutex()
        var api: TessBaseAPI? = null
        var langs: String? = null
    }

    // Two slots on >=4-core devices: page recognition runs one-per-core-pair,
    // hiding much of a page's latency behind the next page's.
    private val slots =
        List(if (Runtime.getRuntime().availableProcessors() >= 4) 2 else 1) { EngineSlot() }

    /** Registered from `MyApplication.onCreate`; the engine's only Context source. */
    fun registerAppContext(context: Context) {
        appContext = context.applicationContext
    }

    /** Mirrors ML Kit init: records the script and warms the engine in the background. */
    fun init(language: OcrLanguage) {
        requestedLangs = resolveTesseractLangs(language)
        ioScope.launch {
            try {
                slots.first().gate.withLock { prepareSlotLocked(slots.first()) }
            } catch (e: Exception) {
                Log.e(TAG, "Tesseract warm-up failed", e)
                Timber.e(e, "Tesseract warm-up failed")
            }
        }
    }

    suspend fun extractTextFromBitmap(
        bitmap: Bitmap,
        onModelDownloading: () -> Unit
    ): OcrResult? {
        // Models ship inside the APK: nothing downloads, so onModelDownloading
        // is deliberately not fired (the reader UI auto-clears its indicator).
        if (bitmap.width <= 0 || bitmap.height <= 0) return null
        return withContext(Dispatchers.IO) {
            try {
                val free = slots.firstOrNull { it.gate.tryLock() }
                if (free != null) {
                    try {
                        ocrOnSlot(free, bitmap)
                    } finally {
                        free.gate.unlock()
                    }
                } else {
                    // Every slot busy: wait on the first instead of queueing.
                    slots.first().gate.withLock { ocrOnSlot(slots.first(), bitmap) }
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "Tesseract OCR failed", e)
                Timber.e(e, "Tesseract OCR failed")
                null
            }
        }
    }

    /** Releases every engine instance. Safe to call repeatedly and before init. */
    fun release() {
        ioScope.launch {
            slots.forEach { slot ->
                slot.gate.withLock {
                    slot.api?.recycle()
                    slot.api = null
                    slot.langs = null
                }
            }
        }
    }

    /** Runs one recognition on an already-locked [slot]. */
    private fun ocrOnSlot(slot: EngineSlot, bitmap: Bitmap): OcrResult? {
        val tess = prepareSlotLocked(slot) ?: return null
        return ocrBitmap(tess, bitmap)
    }

    /** Returns the usable engine for [slot] or null; caller must hold slot's gate. */
    private fun prepareSlotLocked(slot: EngineSlot): TessBaseAPI? {
        val langs = requestedLangs
        slot.api?.let { existing ->
            if (slot.langs == langs) return existing
            existing.recycle()
            slot.api = null
            slot.langs = null
        }

        val context = appContext ?: run {
            Log.w(TAG, "Tesseract skipped: no application context registered")
            Timber.w("Tesseract skipped: no application context registered")
            return null
        }
        val dataPath = ensureTessdata(context, langs) ?: run {
            Log.w(TAG, "Tesseract skipped: tessdata files unavailable for $langs")
            Timber.w("Tesseract skipped: tessdata files unavailable for $langs")
            return null
        }

        val started = TessBaseAPI()
        if (!started.init(dataPath, langs)) {
            // Do not delete traineddata on failure: a transient init error would
            // otherwise wipe the models and break every later OCR call. Just
            // recycle and let ensureTessdata re-validate on the next attempt.
            started.recycle()
            Log.e(TAG, "Tesseract init failed for $langs")
            Timber.e("Tesseract init failed for $langs")
            return null
        }
        // Scan as a full page (handles sparse pages and mixed layouts).
        started.setPageSegMode(TessBaseAPI.PageSegMode.PSM_AUTO)
        // Preserve word spacing in the recognized text (helps Persian search)
        // and tell Tesseract the approximate DPI of the rendered page.
        started.setVariable("preserve_interword_spaces", "1")
        started.setVariable("user_defined_dpi", "300")
        slot.api = started
        slot.langs = langs
        return started
    }

    private fun resolveTesseractLangs(language: OcrLanguage): String = when (language) {
        OcrLanguage.PERSIAN -> LANGS_PERSIAN_PLUS_ENGLISH
        OcrLanguage.PERSIAN_FAST -> LANGS_PERSIAN_ONLY
        // Only Persian and English models ship in the OSS build; other script
        // choices fall back to the English model rather than failing.
        else -> LANGS_FALLBACK
    }

    /** Copies bundled traineddata into filesDir on first run; returns the init data path. */
    private fun ensureTessdata(context: Context, langs: String): String? {
        val destDir = File(context.filesDir, TESSDATA_DIR)
        if (!destDir.exists()) destDir.mkdirs()

        val bundled = try {
            context.assets.list(TESSDATA_DIR).orEmpty().filter { it.endsWith(".traineddata") }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to list bundled tessdata", e)
            Timber.e(e, "Failed to list bundled tessdata")
            emptyList()
        }

        for (name in bundled) {
            val target = File(destDir, name)
            if (target.length() < MIN_TRAINEDDATA_BYTES) {
                try {
                    // Unique suffix: two engine slots can copy concurrently on first run.
                    val tmp = File(destDir, "$name.tmp-${System.nanoTime()}")
                    context.assets.open("$TESSDATA_DIR/$name").use { input ->
                        tmp.outputStream().use { input.copyTo(it) }
                    }
                    if (tmp.length() < MIN_TRAINEDDATA_BYTES || !tmp.renameTo(target)) {
                        tmp.delete()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to copy $name into filesDir", e)
                    Timber.e(e, "Failed to copy $name into filesDir")
                }
            }
        }

        val langsReady = langs.split('+')
            .all { lang -> File(destDir, "$lang.traineddata").length() >= MIN_TRAINEDDATA_BYTES }
        return if (bundled.isNotEmpty() && langsReady) context.filesDir.absolutePath else null
    }

    private class ProcessedBitmap(val bitmap: Bitmap, val scaleX: Float, val scaleY: Float)

    /** Renders one bitmap through the engine and returns ML Kit-shaped results. */
    private fun ocrBitmap(tess: TessBaseAPI, input: Bitmap): OcrResult {
        val prepared = prepareForRecognition(input)
        try {
            tess.setImage(prepared.bitmap)
            // getUTF8Text returns page text in logical order: for `fas` that is
            // RTL, matching what the shared search regex and selection code
            // expect, and Compose bidi-renders it as written.
            val text = tess.getUTF8Text().orEmpty()
            val result = tess.resultIterator?.let { iterator: ResultIterator ->
                try {
                    collectResults(iterator, text, prepared, input.width.toFloat(), input.height.toFloat())
                } finally {
                    iterator.delete()
                }
            }
            return result ?: OcrResult(text = text, textBlocks = emptyList())
        } finally {
            tess.clear()
            prepared.bitmap.recycle()
        }
    }

    /**
     * Downscale + grayscale in a single draw pass; scales are what maps Tesseract
     * boxes back into the caller's coordinate space.
     */
    private fun prepareForRecognition(input: Bitmap): ProcessedBitmap {
        val bitmap = if (input.config == null || input.config == Bitmap.Config.HARDWARE) {
            input.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            input
        }

        val targetWidth = min(bitmap.width, MAX_OCR_WIDTH)
        val targetHeight = (targetWidth.toLong() * bitmap.height / bitmap.width).toInt().coerceAtLeast(1)
        val gray = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(gray)
        canvas.drawColor(Color.WHITE)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(0f) })
        }
        canvas.drawBitmap(
            bitmap,
            null,
            RectF(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat()),
            paint
        )
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        if (bitmap !== input) bitmap.recycle()
        return ProcessedBitmap(
            bitmap = gray,
            scaleX = sourceWidth.toFloat() / targetWidth,
            scaleY = sourceHeight.toFloat() / targetHeight
        )
    }

    /**
     * Walks the result iterator at symbol level and regroups the words/lines/
     * blocks like ML Kit exposes them. Tesseract gives integer boxes in the
     * processed bitmap space, so every box is scaled back into the input space.
     */
    private fun collectResults(
        iterator: ResultIterator,
        text: String,
        prepared: ProcessedBitmap,
        inputWidth: Float,
        inputHeight: Float
    ): OcrResult {
        iterator.begin()
        // After begin() the position sits before the first object; advance to
        // an actual symbol start before reading anything.
        if (!iterator.isAtBeginningOf(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL) &&
            !iterator.next(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)
        ) {
            return OcrResult(text = text, textBlocks = emptyList())
        }

        val blocks = mutableListOf<OcrBlock>()
        val linesInBlock = mutableListOf<OcrLine>()
        var wordSymbols = mutableListOf<OcrSymbol>()
        var lineElements = mutableListOf<OcrElement>()
        var first = true

        fun flushWord() {
            if (wordSymbols.isEmpty()) return
            val box = unionBox(wordSymbols.mapNotNull { it.boundingBox })
            lineElements.add(
                OcrElement(
                    text = wordSymbols.joinToString(separator = "") { it.text },
                    boundingBox = box?.let { mapBox(it, prepared, inputWidth, inputHeight) },
                    symbols = wordSymbols
                )
            )
            wordSymbols = mutableListOf()
        }

        fun flushLine() {
            flushWord()
            if (lineElements.isEmpty()) return
            val box = unionBox(lineElements.mapNotNull { it.boundingBox })
            linesInBlock.add(
                OcrLine(
                    text = lineElements.joinToString(separator = " ") { it.text },
                    boundingBox = box?.let { mapBox(it, prepared, inputWidth, inputHeight) },
                    elements = lineElements
                )
            )
            lineElements = mutableListOf()
        }

        fun flushBlock() {
            flushLine()
            if (linesInBlock.isEmpty()) return
            val box = unionBox(linesInBlock.mapNotNull { it.boundingBox })
            blocks.add(
                OcrBlock(
                    text = linesInBlock.joinToString(separator = "\n") { it.text },
                    boundingBox = box?.let { mapBox(it, prepared, inputWidth, inputHeight) },
                    lines = linesInBlock.toList()
                )
            )
            linesInBlock.clear()
        }

        while (true) {
            val startsBlock = first || iterator.isAtBeginningOf(TessBaseAPI.PageIteratorLevel.RIL_BLOCK)
            val startsLine = startsBlock || iterator.isAtBeginningOf(TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE)
            val startsWord = startsLine || iterator.isAtBeginningOf(TessBaseAPI.PageIteratorLevel.RIL_WORD)

            if (startsBlock) {
                flushBlock()
            } else if (startsLine) {
                flushLine()
            } else if (startsWord) {
                flushWord()
            }

            val symbolText = iterator.getUTF8Text(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)
            val symbolBox = iterator.getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)

            if (!symbolText.isNullOrEmpty() && symbolBox != null && !symbolBox.isEmpty) {
                wordSymbols.add(
                    OcrSymbol(
                        text = symbolText,
                        boundingBox = mapBox(symbolBox, prepared, inputWidth, inputHeight)
                    )
                )
                first = false
            }

            if (!iterator.next(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)) break
        }
        flushBlock()

        // A recognized page always yields at least its raw text even when the
        // iterator produced no usable boxes.
        if (blocks.isEmpty()) {
            return OcrResult(text = text, textBlocks = emptyList())
        }
        return OcrResult(text = text, textBlocks = blocks)
    }

    private fun mapBox(box: Rect, prepared: ProcessedBitmap, inputWidth: Float, inputHeight: Float): Rect {
        val left = (box.left * prepared.scaleX).roundToInt().coerceAtLeast(0)
        val top = (box.top * prepared.scaleY).roundToInt().coerceAtLeast(0)
        val right = (box.right * prepared.scaleX).roundToInt().coerceAtMost(inputWidth.roundToInt())
        val bottom = (box.bottom * prepared.scaleY).roundToInt().coerceAtMost(inputHeight.roundToInt())
        return Rect(left, top, right.coerceAtLeast(left), bottom.coerceAtLeast(top))
    }

    private fun unionBox(rects: List<Rect>): Rect? {
        if (rects.isEmpty()) return null
        val out = Rect(rects.first())
        rects.drop(1).forEach { out.union(it) }
        return out
    }
}
