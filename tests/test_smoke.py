"""Smoke tests: 验证包能正常 import + 核心函数可调用。

跑法：
    pytest tests/
    pytest tests/ -v
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# 1. 静态 import 测试
# ---------------------------------------------------------------------------
class TestImports:
    """所有公开模块都能正常 import。"""

    def test_routing_imports(self):
        from abexp.routing import (
            ab_split_validator,
            realtime_prebucket,
            realtime_remedy,
            realtime_breakthrough,
            realtime_adaptive,
            calibration,
            orthogonal_layers,
            bucket_count_analysis,
            streaming_vs_batch,
            bias_vs_traffic,
        )
        assert all(m is not None for m in [
            ab_split_validator, realtime_prebucket, realtime_remedy,
            realtime_breakthrough, realtime_adaptive, calibration,
            orthogonal_layers, bucket_count_analysis, streaming_vs_batch,
            bias_vs_traffic,
        ])

    def test_analysis_imports(self):
        from abexp.analysis import (
            did_cuped_analysis, did_cuped_kaggle, did_cuped_consumption,
            beta_binomial, outlier_handling,
        )
        assert all(m is not None for m in [
            did_cuped_analysis, did_cuped_kaggle, did_cuped_consumption,
            beta_binomial, outlier_handling,
        ])

    def test_validation_imports(self):
        from abexp.validation import (
            experiment_validation_report, aa_test,
            seasonal_early_stop, full_scale_validation,
        )
        assert all(m is not None for m in [
            experiment_validation_report, aa_test,
            seasonal_early_stop, full_scale_validation,
        ])

    def test_advanced_imports(self):
        from abexp.advanced import ab_rampup_strategy
        assert ab_rampup_strategy is not None

    def test_tools_imports(self):
        from abexp.tools import generate_test_data, sample_size_table
        assert generate_test_data is not None
        assert sample_size_table is not None


# ---------------------------------------------------------------------------
# 2. 核心函数行为测试
# ---------------------------------------------------------------------------
class TestAbSplitValidator:
    """核心分流算法的小规模行为测试。"""

    def test_snake_assignment_basic(self):
        from abexp.routing.ab_split_validator import assign_groups
        user_ids = [f"u_{i:04d}" for i in range(1000)]
        groups = assign_groups(
            user_ids, num_buckets=100, num_groups=10, salt="test",
        )
        # 1000 用户 / 10 组 -> 每组 100（蛇形允许 ±5% 浮动，符合 √n 下界）
        sizes = [len(v) for v in groups.values()]
        assert sum(sizes) == 1000
        # 蛇形分配理论上应比纯 hash 偏差小；这里允许 5% 浮动
        for s in sizes:
            assert 95 <= s <= 105, f"uneven: {sizes}"

    def test_srm_check_healthy(self):
        from abexp.routing.ab_split_validator import srm_check
        # 100/100/100/100 -> SRM 健康
        result = srm_check([100, 100, 100, 100])
        assert result[-2] is True  # passed
        assert result[1] > 0.5  # 完全均匀时 p 应接近 1

    def test_srm_check_unhealthy(self):
        from abexp.routing.ab_split_validator import srm_check
        # 50/150/100/100 -> SRM 失衡
        result = srm_check([50, 150, 100, 100])
        assert result[-2] is False  # passed
        assert result[1] < 0.05  # p < 0.05

    def test_deterministic_same_seed(self):
        """相同输入应产生相同的分配结果。"""
        from abexp.routing.ab_split_validator import assign_groups
        uids = [f"u_{i:04d}" for i in range(500)]
        g1 = assign_groups(uids, num_buckets=100, num_groups=10, salt="s")
        g2 = assign_groups(uids, num_buckets=100, num_groups=10, salt="s")
        for k in g1:
            assert g1[k] == g2[k]


class TestValidationAPI:
    """实验检验 API 的接口签名前向兼容。"""

    def test_validate_full_pipeline_signature(self):
        from abexp.validation.experiment_validation_report import validate_full_pipeline
        sig = inspect.signature(validate_full_pipeline)
        params = list(sig.parameters.keys())
        for required in ["df", "group_col", "y_col", "feature_cols", "experiment_id"]:
            assert required in params, f"missing parameter: {required}"


class TestPublicApi:
    """公开 API 完整性。"""

    def test_top_level_helpers(self):
        """abexp.__init__ 暴露的主要函数。"""
        from abexp.routing.ab_split_validator import (
            assign_groups,
            srm_check,
            calc_mde,
            NUM_USERS,
            NUM_BUCKETS,
            NUM_GROUPS,
        )
        assert callable(assign_groups)
        assert callable(srm_check)
        assert callable(calc_mde)
        assert NUM_USERS > 0
        assert NUM_BUCKETS > NUM_GROUPS
