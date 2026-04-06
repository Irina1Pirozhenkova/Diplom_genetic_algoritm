# Ethnogenesis CLI

Python-проект для стохастического моделирования этногенеза по мотивам статьи
«МОДЕЛЬ ЭВОЛЮЦИИ ЭТНОСОВ И ЕЕ РЕАЛИЗАЦИЯ НА БАЗЕ ГЕНЕТИЧЕСКИХ АЛГОРИТМОВ».

Итоговый вывод состоит из шести графиков:

- базовая пассионарность;
- пассионарность при вариации `E` в `3` раза вверх и вниз;
- пассионарность при вариации `K` в `3` раза вверх и вниз;
- два шоковых сценария;
- базовый график общесистемного критерия эффективности `C(t)`.

Во всех итоговых экспериментах число призваний фиксировано: `I = 4`.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Если виртуальное окружение не нужно:

```powershell
python -m pip install -e .
```

## Быстрый запуск

Если в каталоге `artifacts` уже лежит `best_config.json`, графики можно перестроить так:

```powershell
python -m ethnogenesis render --config artifacts/best_config.json
```

Будут сохранены файлы:

- `artifacts/passionarity_base.png`
- `artifacts/passionarity_E.png`
- `artifacts/passionarity_K.png`
- `artifacts/shock_1072.png`
- `artifacts/shock_647.png`
- `artifacts/criterion_C_base.png`
- `artifacts/experiment_scenarios.json`

## Поиск параметров

Полный поиск лучшего набора параметров:

```powershell
python -m ethnogenesis search
```

Более быстрый вариант:

```powershell
python -m ethnogenesis search --trials 40 --seeds-per-trial 3
```

После поиска сохраняются файлы:

- `artifacts/best_config.json`
- `artifacts/search_result.json`
- `artifacts/passionarity_base.png`
- `artifacts/passionarity_E.png`
- `artifacts/passionarity_K.png`
- `artifacts/shock_1072.png`
- `artifacts/shock_647.png`
- `artifacts/criterion_C_base.png`
- `artifacts/experiment_scenarios.json`

## Проверка

```powershell
python -m unittest discover -s tests -v
```
