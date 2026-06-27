# DINOv2 Duplicate Image Finder — C# WPF

Найти и отобразить группы дубликатов или близких изображений в папке, используя глубокое обучение (DINOv2) и геометрическую верификацию (SIFT/WGC).

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
dotnet build
dotnet run --project DinoDuplicateSearch.csproj
```

## Зависимости (NuGet)

| Пакет | Назначение |
|-------|-----------|
| `Microsoft.ML.OnnxRuntime.DirectML` | Инференс DINOv2 (GPU через DirectML) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` | SIFT, BFMatcher, обработка изображений |
| `System.Data.SQLite` | Persistent кэш эмбеддингов, SIFT и WGC результатов |

## Структура проекта

```
DinoDuplicateSearch/
├── Core/
│   ├── DuplicatesFinder.cs         # Основная логика: эмбеддинги, кластеризация, WGC, клики
│   ├── FeatureCache.cs             # SQLite кэш: эмбеддинги, SIFT, WGC результаты
│   ├── GeometricConsistency.cs     # SIFT + Weak Geometric Consistency (MinAbsoluteVotes=50)
│   ├── AgglomerativeClustering.cs  # Ручная реализация агломеративной кластеризации
│   └── ImageUtils.cs               # Загрузка изображений (ASCII + Unicode пути)
├── Models/
│   ├── Models.cs                   # DuplicatePair, DuplicateGroup
│   └── UnionFind.cs                # Union-Find (резервный)
├── ViewModels/
│   ├── MainViewModel.cs            # MVVM: навигация, открытие изображений
│   ├── SearchViewModel.cs          # Выбор папки, настройки, фоновый поиск
│   └── ResultsViewModel.cs         # Отображение результатов
├── Views/
│   ├── MainWindow.xaml             # TabControl (Search / Results)
│   ├── SearchView.xaml             # Directory picker, слайдеры, кнопки
│   └── ResultsView.xaml            # ScrollViewer + сетка миниатюр
├── Converters/
│   └── Converters.cs               # IValueConverter для XAML-биндингов
├── export_model.py                 # Скрипт экспорта DINOv2 → ONNX
└── debug.log                       # Логи отладки (создаётся автоматически)
```

## Как это работает

1. **DINOv2 эмбеддинги** — для каждого изображения извлекается 768-мерный CLS-вектор через ONNX Runtime (GPU DirectML)
2. **Агломеративная кластеризация** — изображения группируются по cosine distance
3. **SIFT + WGC** (опционально) — для каждой пары из кластера:
   - Извлекаются SIFT-ключевые точки
   - Находятся соответствия через BFMatcher
   - Проверяется гистограмма углов и масштабов
   - Требуется минимум 50 голосов (MinAbsoluteVotes) для прохождения
4. **Группировка (клики)** — изображения объединяются в группу только если **все пары** между ними прошли WGC (алгоритм Bron-Kerbosch для поиска клик)
5. **Кэширование** — эмбеддинги, SIFT-признаки и результаты WGC кэшируются в SQLite. Инвалидация по mtime файла

## Настройки в UI

| Параметр | Диапазон | По умолчанию | Описание |
|----------|----------|--------------|----------|
| Distance Threshold | 0.01 – 1.0 | 0.45 | Порог кластеризации (ниже = строже) |
| Geometric Verification | Вкл/Выкл | Вкл | SIFT + WGC вторая проверка |
| WGC Threshold | 0.1 – 0.9 | 0.30 | Минимальная доля совпадений для WGC |
| Min similarity for pair | 0.5 – 1.0 | 0.50 | Минимальное схожесть пары для добавления в граф WGC |

## Прогресс

Во время поиска отображается:
- Текущий этап (эмбеддинги / кластеризация / WGC)
- Для WGC: имена файлов, PASS/FAIL, sim, angle votes, scale votes
- Процент выполнения

## Портировано с Python/Kivy

Оригинальный проект: Python + Kivy 2.x + PyTorch + scikit-learn + OpenCV

| Аспект | Python (было) | C# WPF (стало) |
|--------|---------------|-----------------|
| GUI | Kivy 2.x | WPF (.NET 8) |
| ML инференс | PyTorch | ONNX Runtime (DirectML GPU) |
| Кластеризация | scikit-learn | Ручная реализация |
| Группировка | Union-Find (транзитивная) | Bron-Kerbosch (клики — все пары) |
| OpenCV | opencv-python | OpenCvSharp4 |
| Кэш | sqlite3 + zlib | System.Data.SQLite (эмбеддинги + SIFT + WGC) |
| Упаковка | pip + venv | `dotnet publish` |

## Лицензия

Свободное использование.

Для работы GPU могут понадобится опреденные версии cuda и cudnn. Если их нет в path, то можно указать в appsettings.json:
{
  "cuda_path": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin",
  "cudnn_path": "c:\\Program Files\\NVIDIA\\CUDNN\\v9.23\\bin\\12.9\\x64"
} 
