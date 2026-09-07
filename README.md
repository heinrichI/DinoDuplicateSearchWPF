# DINOv2 Duplicate Image Finder — C# WPF

Найти и отобразить группы дубликатов или близких изображений в папке, используя глубокое обучение (DINOv2) и геометрическую верификацию (SuperPoint + LightGlue или SIFT/WGC).

Специально разработана чтобы находить сильно обрезанные изображения.
<img width="1479" height="1036" alt="image" src="https://github.com/user-attachments/assets/8f003d4c-de66-418d-bc9a-76b8bd903821" />

## Требования

- **Windows** 10/11 (x64)
- **.NET 8 SDK** — [скачать](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Python 3.10+** (только для экспорта модели в ONNX)
- **PyTorch + Transformers** (для экспорта модели)

## Быстрый старт

### 1. Скачивание моделей

| Модель | Назначение | Ссылка | Размер |
|--------|-----------|--------|--------|
| **DINOv2-base** | Извлечение 768-D эмбеддингов | [HuggingFace](https://huggingface.co/facebook/dinov2-base) | ~330 МБ |
| **SuperPoint+LightGlue pipeline** | Fused feature extraction + matching | [Скачать](https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v2.0/superpoint_lightglue_pipeline.onnx) | ~51 МБ |

#### DINOv2 (обязательно)

```bash
pip install torch transformers
python export_model.py
```

Создаст файл `Models/dinov2-base.onnx` с **register-токенами** (CLS + 4 registers + 196 patches = 201 токен).

#### SuperPoint + LightGlue (рекомендуется)

Скачайте fused pipeline модель из [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX):
```bash
curl -L -o Models/superpoint_lightglue_pipeline.onnx https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v2.0/superpoint_lightglue_pipeline.onnx
```

> **Примечание:** Если модели SuperPoint/LightGlue не найдены, приложение автоматически использует **SIFT + BFMatcher** (legacy пайплайн).

### 2. Сборка и запуск

```bash
dotnet build src/DinoDuplicateSearch.sln
dotnet run --project src/DinoDuplicateSearch.WPF/DinoDuplicateSearch.WPF.csproj
```

## Зависимости (NuGet)

| Пакет | Назначение |
|-------|-----------|
| `Microsoft.ML.OnnxRuntime.Gpu` | Инференс DINOv2, SuperPoint, LightGlue (GPU через CUDA, fallback на CPU) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` | SIFT, BFMatcher, обработка изображений |
| `System.Data.SQLite` | Persistent кэш эмбеддингов, SIFT и WGC результатов |

## Структура проекта

```
src/
├── DinoDuplicateSearch.Models/
│   ├── Models.cs                   # DuplicatePair, DuplicateGroup, ProgressData
│   ├── SearchSettings.cs           # Настройки поиска (clustering params + LightGlue)
│   ├── UnionFind.cs                # Union-Find (резервный)
│   └── DebugLog.cs                 # Thread-safe логгер
├── DinoDuplicateSearch.Database/
│   └── FeatureCache.cs             # SQLite кэш: эмбеддинги, SIFT, WGC результаты
├── DinoDuplicateSearch.ML/
│   ├── EmbeddingExtractor.cs       # DINOv2 ONNX инференс (GPU/CPU), батчевая обработка
│   └── AgglomerativeClustering.cs  # Трёхрежимная кластеризация + порог транзитивности
├── DinoDuplicateSearch.CV/
│   ├── GeometricConsistency.cs     # SIFT + WGC (legacy) и SuperPoint + LightGlue (новый)
│   ├── SuperPointLightGluePipeline.cs # Fused SuperPoint+LightGlue ONNX pipeline
│   ├── SuperPointExtractor.cs      # SuperPoint ONNX: ключевые точки + 256-D дескрипторы
│   ├── LightGlueMatcher.cs         # LightGlue ONNX: learned feature matching
│   └── ImageUtils.cs               # Загрузка изображений (ASCII + Unicode + WebP fallback)
├── DinoDuplicateSearch.WPF/
│   ├── Core/
│   │   ├── DuplicatesFinder.cs     # Основная логика: эмбеддинги, кластеризация, WGC, клики
│   │   ├── ProductQuantizer.cs     # Product Quantization для быстрого поиска (SIMD + parallel)
│   │   └── ThrottledProgress.cs    # Throttled progress reporter
│   ├── ViewModels/
│   │   ├── MainViewModel.cs        # MVVM: навигация, открытие изображений
│   │   ├── SearchViewModel.cs      # Выбор папки, настройки, фоновый поиск
│   │   └── ResultsViewModel.cs     # Отображение результатов
│   ├── Views/
│   │   ├── MainWindow.xaml         # TabControl (Search / Results)
│   │   ├── SearchView.xaml         # Directory picker, слайдеры с tooltip, кнопки
│   │   └── ResultsView.xaml        # ScrollViewer + сетка миниатюр
│   └── Converters/
│       └── Converters.cs           # IValueConverter для XAML-биндингов
├── export_model.py                 # Скрипт экспорта DINOv2 → ONNX (с register-токенами)
├── export_superpoint.py            # Скрипт экспорта SuperPoint → ONNX
├── export_lightglue.py             # Скрипт экспорта LightGlue → ONNX
└── README.md
```

## Как это работает

1. **DINOv2 эмбеддинги** — для каждого изображения извлекается 768-мерный CLS-вектор через ONNX Runtime (GPU CUDA, fallback на CPU). Эмбеддинги нормализуются (L2) для косинусного сравнения. Модель с **register-токенами** для устранения outlier tokens.

2. **Кластеризация** — трёхрежимный алгоритм с **порогом транзитивности**:
   - **≤ 10 000 изображений**: агломеративная кластеризация (средняя связь) с полной матрицей расстояний
   - **10 000 – 50 000 изображений**: Union-Find с попарным сравнением на лету, SIMD-оптимизированное скалярное произведение (`System.Numerics.Vector<float>`), параллельная обработка (`Parallel.For`). Потребление памяти O(n) вместо O(n²)
   - **> 50 000 изображений**: Product Quantization (PQ) — обучение 96 подпространств по 256 центроидов, поиск top-5 ближайших соседей через ADC-таблицу, кластеризация через Union-Find. SIMD + параллелизм на всех этапах. Кэширование центроидов и результатов поиска в файлы с инвалидацией по хешу эмбеддингов
   - **Порог транзитивности**: рёбра сортируются по расстоянию. Плотные связи (dist < ratio × threshold) объединяются безусловно. Слабые связи (dist ≥ ratio × threshold) объединяются только для маленьких кластеров (≤50). Это разрывает цепочки A~B~C и предотвращает гигантские кластеры
   - **Ограничение размера кластера**: кластеры > MaxClusterSize разбиваются на одиночные изображения

3. **Геометрическая верификация** (опционально) — для каждой пары из кластера (параллельно через `Parallel.ForEach`) выбирается один из двух бэкендов:
   - **SuperPoint + LightGlue (fused, по умолчанию)** — одна ONNX модель из [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) — извлечение ключевых точек + learned matching. Устойчива к кропу, вращению, освещению
   - **SIFT + BFMatcher (классический)** — признаки OpenCV SIFT + проверка геометрической согласованности. Используется только при выборе режима «SIFT»
   - Результаты кэшируются в SQLite с проверкой mtime

### Алгоритм: вся работа программы

Полный конвейер поиска дубликатов состоит из 4 этапов: получение хешей (эмбеддингов), кластеризация, геометрическая верификация, группировка в клики.

**1. Хеши (эмбеддинги)** — под «хешем» понимается вектор эмбеддинга, получаемый нейросетью DINOv2.
- Модель `dinov2-base.onnx` (DINOv2 ViT-Base, 768-мерные векторы) выполняется через **ONNX Runtime** на **GPU (CUDA)** с автоматическим fallback на **CPU**, если GPU недоступен.
- Изображение читается через OpenCV, конвертируется **BGR → RGB**, масштабируется до **224×224**; пиксели нормируются в интервал `[0, 1]`.
- Изображения объединяются в батчи (по умолчанию **32**) и подаются в модель как тензор `(N, 3, 224, 224)` (вход `pixel_values`).
- Модель возвращает `last_hidden_state` формы `(batch, 201, 768)` — это **201 токен**: **CLS-токен (1) + register-токены (4) + токены патчей (196)**. В качестве хеша берётся только **первый (CLS) токен** — вектор размерности 768.
- Вектор нормализуется по **L2**, чтобы сравнение пар сводилось к косинусной близости (скалярное произведение нормированных векторов), и кэшируется в **SQLite** по `mtime` файла.

**2. Кластеризация** — поиск пар близких по хешу изображений с порогом **Distance Threshold** (по умолчанию 0.45):
- сравнение через **скалярное произведение** L2-нормированных векторов;
- для больших наборов (**> 50 000**) векторы сжимаются через **Product Quantization (PQ)** — 96 подпространств × 256 центроидов — вместо попарного сравнения всех пар (SIMD + параллельный k-means, поиск top-5 соседей);
- применяется **порог транзитивности** (по умолчанию 0.70) и ограничение размера кластера (MaxClusterSize), чтобы рвать цепочки `A~B~C`.

**3. Геометрическая верификация** (опционально) — для каждой пары из кластера (параллельно через `Parallel.ForEach`) выбирается один из двух бэкендов. Автоматического fallback между ними нет:
- **SuperPoint + LightGlue (по умолчанию)** — fused-модель `superpoint_lightglue_pipeline.onnx` (fabio-sim/LightGlue-ONNX) за один проход извлекает ключевые точки (SuperPoint) и выполняет learned-матчинг (LightGlue). Изображения масштабируются к размеру 800×600 (кратному 8) и дополняются до общего размера для входа `(2, 1, H, W)`, координаты возвращаются к исходному масштабу. Пара проходит проверку, если одновременно: **≥ 10 матчей**, **≥ 5 голосов в одном бине гистограммы углов (24 бина)** и **средняя уверенность ≥ `Min match score`** (по умолчанию 0.50).
- **SIFT + WGC** — классический пайплайн OpenCV: извлечение ключевых точек и дескрипторов SIFT (`SiftGeometricVerifier`), поиск соответствий через BFMatcher и проверка геометрической согласованности (WGC, гистограмма углов/масштабов). Используется только при ручном выборе режима «SIFT»; порог — `WGC Threshold` (по умолчанию 0.30).
- Если включён LightGlue, SIFT не вызывается; при выключенной опции **«Enable geometric verification»** ни один бэкенд не вызывается — пары объединяются только по кластеризации эмбеддингов. Результаты верификации кэшируются в SQLite с проверкой mtime.

**4. Группировка в клики** — изображения объединяются в группу только если **все пары** между ними прошли верификацию (алгоритм **Bron-Kerbosch** для поиска клик в графе проверенных пар).

4. **Группировка (клики)** — изображения объединяются в группу только если **все пары** между ними прошли верификацию (алгоритм Bron-Kerbosch для поиска клик)

5. **Кэширование** — эмбеддинги, SIFT-признаки и результаты WGC кэшируются в SQLite (Zlib-сжатие, WAL режим, busy_timeout=5000). Центроиды PQ и результаты поиска кэшируются в бинарные файлы с инвалидацией по хешу эмбеддингов

## Настройки в UI

| Параметр | Диапазон | По умолчанию | Описание |
|----------|----------|--------------|----------|
| Distance Threshold | 0.01 – 1.0 | 0.45 | Порог кластеризации (ниже = строже) |
| Geometric Verification | Вкл/Выкл | Вкл | Геометрическая верификация |
| Use LightGlue | Вкл/Выкл | Вкл | SuperPoint+LightGlue (выкл = SIFT fallback) |
| Search Subfolders | Вкл/Выкл | Выкл | Рекурсивный поиск в подпапках |
| WGC Threshold | 0.1 – 0.9 | 0.30 | Минимальная доля совпадений для верификации |
| Min Similarity for Pair | 0.5 – 1.0 | 0.50 | Минимальная схожесть пары для добавления в граф |
| Batch Size | 1 – 512 | 32 | Количество изображений в батче ONNX |
| Prefetch | 0 – 2560 | 2 | Глубина конвейера producer/consumer |
| Max Cluster Size | 10 – 1000 | 200 | Макс. изображений в кластере |
| Transitivity Ratio | 0.1 – 1.0 | 0.70 | Порог транзитивности (ниже = строже) |
| SuperPoint Max Keypoints | 100 – 5000 | 2000 | Макс. ключевых точек SuperPoint |
| LightGlue Confidence | 0.1 – 1.0 | 0.50 | Мин. уверенность матчей LightGlue |

## Прогресс

Во время поиска отображается:
- Текущий этап (эмбеддинги / кластеризация / WGC)
- Прогресс PQ: `k-means: 5 из 96 подпространств`
- Прогресс WGC: `WGC: 42/5229 file1 vs file2 [PASS] sim=0.630`
- Процент выполнения

## Портировано с Python/Kivy

Оригинальный проект: Python + Kivy 2.x + PyTorch + scikit-learn + OpenCV

| Аспект | Python (было) | C# WPF (стало) |
|--------|---------------|-----------------|
| GUI | Kivy 2.x | WPF (.NET 8) |
| ML инференс | PyTorch | ONNX Runtime (CUDA GPU) |
| Кластеризация | scikit-learn | Трёхрежимная + порог транзитивности + ограничение размера |
| Группировка | Union-Find (транзитивная) | Bron-Kerbosch (клики — все пары) |
| OpenCV | opencv-python | OpenCvSharp4 |
| Кэш | sqlite3 + zlib | SQLite (WAL + busy_timeout) + PQ бинарные файлы |
| Упаковка | pip + venv | `dotnet publish` |

## Лицензия

Свободное использование.

## GPU

Для работы GPU могут понадобиться определенные версии CUDA и cuDNN. Если их нет в PATH, то можно указать в `appsettings.json`:

```json
{
  "cuda_path": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin",
  "cudnn_path": "C:\\Program Files\\NVIDIA\\CUDNN\\v9.23\\bin\\12.9\\x64"
}
```
