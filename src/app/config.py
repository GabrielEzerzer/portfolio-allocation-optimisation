"""
Configuration management for ACO Portfolio Optimizer.
Loads YAML configuration with environment variable substitution.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _substitute_env_vars(value: str) -> str:
    """Substitute ${VAR:default} patterns with environment variables."""
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ''
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replacer, value)


def _process_config_values(obj: Any) -> Any:
    """Recursively process config values to substitute environment variables."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    return obj


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    base_url: str = ""
    api_key: str = ""
    rate_limit_per_minute: int = 5
    timeout_seconds: int = 30
    retries: int = 3
    backoff_base: float = 2.0
    enabled: bool = True


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    enabled: bool = True
    timeout_seconds: int = 60
    fallback_file: str = ""
    history_days: int = 365
    indicators: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    path: str = "data/cache"
    refresh_days: int = 1
    format: str = "parquet"


@dataclass
class ACOConfig:
    """ACO optimizer configuration."""
    num_ants: int = 20
    num_iterations: int = 50
    evaporation_rate: float = 0.3
    alpha: float = 1.0
    beta: float = 2.0
    weight_granularity: float = 0.05
    random_seed: int | None = None
    heuristic_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.3,
        "returns": 0.3,
        "inverse_volatility": 0.2,
        "fundamentals_score": 0.2
    })


@dataclass
class ConstraintsConfig:
    """Portfolio constraint configuration."""
    max_weight_per_ticker: float = 0.10
    min_weight_threshold: float = 0.01
    min_holdings: int = 5
    max_holdings: int = 50
    sum_weights_tolerance: float = 0.001
    sector_cap: float | None = None


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    train_window_days: int = 252
    test_window_days: int = 21
    rebalance_frequency: str = "monthly"
    risk_free_rate: float = 0.0
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    baselines: list[str] = field(default_factory=lambda: ["equal_weight", "spy_benchmark"])


@dataclass
class FeaturesConfig:
    """Feature handling configuration."""
    missing_value_strategy: str = "fill_median"
    required_features: list[str] = field(default_factory=lambda: ["returns_1m", "volatility_21d"])
    normalize: bool = True
    normalization_method: str = "zscore"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/app.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class OutputConfig:
    """Output configuration."""
    path: str = "data/outputs"
    save_weights: bool = True
    save_metrics: bool = True
    save_diagnostics: bool = True
    format: str = "json"


@dataclass
class Config:
    """Main configuration container."""
    mode: str = "live"
    primary_provider: str = "alphavantage"
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    cache: CacheConfig = field(default_factory=CacheConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # CLI overrides
    tickers: list[str] | None = None
    universe_size: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    random_seed: int | None = None


def load_config(config_path: str, local_path: str | None = None) -> Config:
    """
    Load configuration from YAML files.
    
    Args:
        config_path: Path to main config file (e.g., settings.example.yaml)
        local_path: Optional path to local overrides (e.g., settings.local.yaml)
    
    Returns:
        Fully populated Config object
    """
    # Load main config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Load local overrides if present
    if local_path:
        local_file = Path(local_path)
        if local_file.exists():
            with open(local_file, 'r') as f:
                local_config = yaml.safe_load(f)
            raw_config = _deep_merge(raw_config, local_config)
    
    # Try auto-loading local config
    if local_path is None:
        auto_local = config_file.parent / "settings.local.yaml"
        if auto_local.exists():
            with open(auto_local, 'r') as f:
                local_config = yaml.safe_load(f)
            raw_config = _deep_merge(raw_config, local_config)
    
    # Substitute environment variables
    raw_config = _process_config_values(raw_config)
    
    # Build Config object
    return _build_config(raw_config)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_config(raw: dict) -> Config:
    """Build a Config object from raw dictionary."""
    config = Config()
    
    config.mode = raw.get('mode', 'live')
    
    # Providers
    providers_raw = raw.get('providers', {})
    config.primary_provider = providers_raw.get('primary', 'alphavantage')
    
    for name in ['alphavantage', 'yfinance']:
        if name in providers_raw:
            p = providers_raw[name]
            config.providers[name] = ProviderConfig(
                base_url=p.get('base_url', ''),
                api_key=p.get('api_key', ''),
                rate_limit_per_minute=p.get('rate_limit_per_minute', 5),
                timeout_seconds=p.get('timeout_seconds', 30),
                retries=p.get('retries', 3),
                backoff_base=p.get('backoff_base', 2.0),
                enabled=p.get('enabled', True)
            )
    
    # Agents
    agents_raw = raw.get('agents', {})
    for name in ['universe', 'price', 'technical', 'fundamentals', 'sentiment']:
        if name in agents_raw:
            a = agents_raw[name]
            config.agents[name] = AgentConfig(
                enabled=a.get('enabled', True),
                timeout_seconds=a.get('timeout_seconds', 60),
                fallback_file=a.get('fallback_file', ''),
                history_days=a.get('history_days', 365),
                indicators=a.get('indicators', []),
                fields=a.get('fields', [])
            )
    
    # Cache
    cache_raw = raw.get('cache', {})
    config.cache = CacheConfig(
        enabled=cache_raw.get('enabled', True),
        path=cache_raw.get('path', 'data/cache'),
        refresh_days=cache_raw.get('refresh_days', 1),
        format=cache_raw.get('format', 'parquet')
    )
    
    # ACO
    aco_raw = raw.get('aco', {})
    config.aco = ACOConfig(
        num_ants=aco_raw.get('num_ants', 20),
        num_iterations=aco_raw.get('num_iterations', 50),
        evaporation_rate=aco_raw.get('evaporation_rate', 0.3),
        alpha=aco_raw.get('alpha', 1.0),
        beta=aco_raw.get('beta', 2.0),
        weight_granularity=aco_raw.get('weight_granularity', 0.05),
        random_seed=aco_raw.get('random_seed'),
        heuristic_weights=aco_raw.get('heuristic_weights', {})
    )
    
    # Constraints
    constraints_raw = raw.get('constraints', {})
    config.constraints = ConstraintsConfig(
        max_weight_per_ticker=constraints_raw.get('max_weight_per_ticker', 0.10),
        min_weight_threshold=constraints_raw.get('min_weight_threshold', 0.01),
        min_holdings=constraints_raw.get('min_holdings', 5),
        max_holdings=constraints_raw.get('max_holdings', 50),
        sum_weights_tolerance=constraints_raw.get('sum_weights_tolerance', 0.001),
        sector_cap=constraints_raw.get('sector_cap')
    )
    
    # Backtest
    backtest_raw = raw.get('backtest', {})
    config.backtest = BacktestConfig(
        train_window_days=backtest_raw.get('train_window_days', 252),
        test_window_days=backtest_raw.get('test_window_days', 21),
        rebalance_frequency=backtest_raw.get('rebalance_frequency', 'monthly'),
        risk_free_rate=backtest_raw.get('risk_free_rate', 0.0),
        initial_capital=backtest_raw.get('initial_capital', 100000.0),
        transaction_cost=backtest_raw.get('transaction_cost', 0.001),
        baselines=backtest_raw.get('baselines', ['equal_weight', 'spy_benchmark'])
    )
    
    # Features
    features_raw = raw.get('features', {})
    config.features = FeaturesConfig(
        missing_value_strategy=features_raw.get('missing_value_strategy', 'fill_median'),
        required_features=features_raw.get('required_features', ['returns_1m', 'volatility_21d']),
        normalize=features_raw.get('normalize', True),
        normalization_method=features_raw.get('normalization_method', 'zscore')
    )
    
    # Logging
    logging_raw = raw.get('logging', {})
    config.logging = LoggingConfig(
        level=logging_raw.get('level', 'INFO'),
        file=logging_raw.get('file', 'logs/app.log'),
        format=logging_raw.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        max_bytes=logging_raw.get('max_bytes', 10485760),
        backup_count=logging_raw.get('backup_count', 5)
    )
    
    # Output
    output_raw = raw.get('output', {})
    config.output = OutputConfig(
        path=output_raw.get('path', 'data/outputs'),
        save_weights=output_raw.get('save_weights', True),
        save_metrics=output_raw.get('save_metrics', True),
        save_diagnostics=output_raw.get('save_diagnostics', True),
        format=output_raw.get('format', 'json')
    )
    
    return config
