# DINOv2 Duplicate Image Finder — C# WPF

Найти и отобразить группы дубликатов или близких изображений в папке, используя глубокое обучение (DINOv2) и геометрическую верификацию (SIFT/WGC).

Специально разработана чтобы находить сильно обрезанные изображения.
<img width="1479" height="1036" alt="image" src="https://github.com/user-attachments/assets/8f003d4c-de66-418d-bc9a-76b8bd903821" />

## Требования

- **Windows** 10/11 (x64)
- **.NET 8 SDK** — [скачать](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Python 3.10+** (только для экспорта модели в ONNX)
- **PyTorch + Transformers** (для экспорта модели)

## Быстрый старт

### 1. Экспорт модели DINOv2 в ONNX

```bash
pip install torch transformers
cd DinoDuplicateSearch
python export_model.py
```

Создаст файл `Models/dinov2-base.onnx` (~330 МБ).

### 2. Сборка и запуск

```bash
dotnet build src/DinoDuplicateSearch.sln
dotnet run --project src/DinoDuplicateSearch.WPF/DinoDuplicateSearch.WPF.csproj
```

## Зависимости (NuGet)

| Пакет | Назначение |
|-------|-----------|
| `Microsoft.ML.OnnxRuntime.Gpu` | Инференс DINOv2 (GPU через CUDA, fallback на CPU) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` | SIFT, BFMatcher, обработка изображений |
| `System.Data.SQLite` | Persistent кэш эмбеддингов, SIFT и WGC результатов |

## Структура проекта

```
src/
├── DinoDuplicateSearch.Models/
│   ├── Models.cs                   # DuplicatePair, DuplicateGroup, ProgressData
│   ├── SearchSettings.cs           # Настройки поиска (含 clustering params)
│   ├── UnionFind.cs                # Union-Find (резервный)
│   └── DebugLog.cs                 # Thread-safe логгер
├── DinoDuplicateSearch.Database/
│   └── FeatureCache.cs             # SQLite кэш: эмбеддинги, SIFT, WGC результаты
├── DinoDuplicateSearch.ML/
│   ├── EmbeddingExtractor.cs       # DINOv2 ONNX инференс (GPU/CPU), батчевая обработка
│   └── AgglomerativeClustering.cs  # Трёхрежимная кластеризация + порог транзитивности
├── DinoDuplicateSearch.CV/
│   ├── GeometricConsistency.cs     # SIFT + Weak Geometric Consistency
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
├── export_model.py                 # Скрипт экспорта DINOv2 → ONNX
└── README.md
```

## Как это работает

1. **DINOv2 эмбеддинги** — для каждого изображения извлекается 768-мерный CLS-вектор через ONNX Runtime (GPU CUDA, fallback на CPU). Эмбеддинги нормализуются (L2) для косинусного сравнения
2. **Кластеризация** — трёхрежимный алгоритм с **порогом транзитивности**:
   - **≤ 10 000 изображений**: агломеративная кластеризация (средняя связь) с полной матрицей расстояний
   - **10 000 – 50 000 изображений**: Union-Find с попарным сравнением на лету, SIMD-оптимизированное скалярное произведение (`System.Numerics.Vector<float>`), параллельная обработка (`Parallel.For`). Потребление памяти O(n) вместо O(n²)
   - **> 50 000 изображений**: Product Quantization (PQ) — обучение 96 подпространств по 256 центроидов, поиск top-5 ближайших соседей через ADC-таблицу, кластеризация через Union-Find. SIMD + параллелизм на всех этапах. Кэширование центроидов и результатов поиска в файлы с инвалидацией по хешу эмбеддингов
   - **Порог транзитивности**: рёбра сортируются по расстоянию. Плотные связи (dist < ratio × threshold) объединяются безусловно. Слабые связи (dist ≥ ratio × threshold) объединяются только для маленьких кластеров (≤50). Это разрывает цепочки A~B~C и предотвращает гигантские кластеры
   - **Ограничение размера кластера**: кластеры > MaxClusterSize разбиваются на одиночные изображения
3. **SIFT + WGC** (опционально) — для каждой пары из кластера (параллельно через `Parallel.ForEach`):
   - Извлекаются SIFT-ключевые точки
   - Находятся соответствия через BFMatcher
   - Проверяется гистограмма углов и масштабов
   - Требуется минимум 50 голосов (MinAbsoluteVotes) для прохождения
   - Результаты кэшируются в SQLite с проверкой mtime
4. **Группировка (клики)** — изображения объединяются в группу только если **все пары** между ними прошли WGC (алгоритм Bron-Kerbosch для поиска клик)
5. **Кэширование** — эмбеддинги, SIFT-признаки и результаты WGC кэшируются в SQLite (Zlib-сжатие, WAL режим, busy_timeout=5000). Центроиды PQ и результаты поиска邻居 кэшируются в бинарные файлы с инвалидацией по хешу эмбеддингов

## Настройки в UI

| Параметр | Диапазон | По умолчанию | Описание |
|----------|----------|--------------|----------|
| Distance Threshold | 0.01 – 1.0 | 0.45 | Порог кластеризации (ниже = строже) |
| Geometric Verification | Вкл/Выкл | Вкл | SIFT + WGC вторая проверка |
| Search Subfolders | Вкл/Выкл | Выкл | Рекурсивный поиск в подпапках |
| WGC Threshold | 0.1 – 0.9 | 0.30 | Минимальная доля совпадений для WGC |
| Min Similarity for Pair | 0.5 – 1.0 | 0.50 | Минимальная схожесть пары для добавления в граф WGC |
| Batch Size | 1 – 512 | 32 | Количество изображений в батче ONNX |
| Prefetch | 0 – 2560 | 2 | Глубина конвейера producer/consumer |
| Max Cluster Size | 10 – 1000 | 200 | Макс. изображений в кластере (больше = больше пар WGC) |
| Transitivity Ratio | 0.1 – 1.0 | 0.70 | Порог транзитивности (ниже = строже, разрывает цепочки) |

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
