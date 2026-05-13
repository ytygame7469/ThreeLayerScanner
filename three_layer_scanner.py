"""
三层扫描系统 v2.0：个股→板块→主线
基于10轮LLM讨论完善：
- 7因子个股强度（含高位钝化、大小票差异）
- 精确轮动涨停检测（轮动度+质量评分）
- 个股→板块信号传导（领先强度+协同因子）
- 多窗口加权（30+90+250天）
- 多层假信号过滤
- 实战交易信号生成
"""
import sys, os, json, math
sys.path.insert(0, r"D:\new_tdx_test\PYPlugins\user")
from tqcenter import tq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# ============================================================
# 配置参数（基于LLM讨论优化）
# ============================================================
CONFIG = {
    # 窗口参数
    "windows": {"short": 30, "medium": 90, "long": 250},
    "weights": {"short": 0.5, "medium": 0.3, "long": 0.2},

    # 因子权重（趋势市/震荡市，基于ADX判断）
    "factor_weights_trend": {
        "price_pos": 0.20, "trend": 0.35, "momentum": 0.25,
        "vol_price": 0.10, "limit_up": 0.05, "atr": 0.03, "turnover": 0.02,
    },
    "factor_weights_range": {
        "price_pos": 0.40, "trend": 0.20, "momentum": 0.15,
        "vol_price": 0.15, "limit_up": 0.05, "atr": 0.03, "turnover": 0.02,
    },

    # 高位钝化参数
    "decay_pct_threshold": 0.85,
    "decay_min_factor": 0.3,
    "stage_return_limit": 3.0,       # 阶段涨幅>300%限制
    "position_cap_after_decay": 15,  # 高位价格位置上限

    # 强势股定义
    "strong_threshold": 60,

    # 轮动检测参数
    "rotation_window": 20,
    "rotation_new_day_gap": 3,
    "rotation_quality_threshold": 0.5,

    # 板块聚合
    "min_sector_stocks": 5,
    "density_threshold": 0.25,

    # 过滤参数
    "filter_position_median_min": 20,
    "filter_position_median_max": 80,
    "filter_skewness_max": 1.5,
    "filter_divergence_reject": 25,

    # 信号等级
    "signal_levels": {
        "S": {"fg_max": 40, "density_min": 0.30, "rotation_min": 0.6, "position": 0.40},
        "A": {"fg_max": 50, "density_min": 0.20, "rotation_min": 0.4, "position": 0.25},
        "B": {"fg_max": 60, "density_min": 0.10, "rotation_min": 0.2, "position": 0.10},
    },
}


# ============================================================
# 第一层：个股强度扫描 v2.0
# ============================================================
class StockStrengthScannerV2:
    """全市场个股强度扫描器（7因子+动态权重+高位钝化）"""

    def __init__(self):
        self.results = {}

    def scan_all(self, top_n: int = 200) -> Dict[str, dict]:
        try:
            if not hasattr(StockStrengthScannerV2, "_tq_init"):
                tq.initialize("SCAN_V2")
                StockStrengthScannerV2._tq_init = True

            all_stocks = tq.get_stock_list(market="all", list_type=0)
            print(f"  全市场 {len(all_stocks)} 只股票，扫描前 {top_n} 只...")

            stocks = all_stocks[:top_n]
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=CONFIG["windows"]["long"] + 30)).strftime("%Y%m%d")

            result = tq.get_market_data(
                field_list=["Close", "Volume", "High", "Low"],
                stock_list=stocks,
                start_time=start, end_time=end,
                period="1d", dividend_type="front"
            )

            if not isinstance(result, dict):
                print("  ❌ 数据获取失败")
                return {}

            close_df = result.get("Close", pd.DataFrame())
            vol_df = result.get("Volume", pd.DataFrame())
            high_df = result.get("High", pd.DataFrame())
            low_df = result.get("Low", pd.DataFrame())

            print(f"  处理 {len(stocks)} 只股票...")
            for i, code in enumerate(stocks):
                if code not in close_df.columns:
                    continue
                close = close_df[code].dropna()
                if len(close) < 40:
                    continue
                vol = vol_df[code].dropna() if code in vol_df.columns else pd.Series()
                high = high_df[code].dropna() if code in high_df.columns else pd.Series()
                low = low_df[code].dropna() if code in low_df.columns else pd.Series()

                score = self._calc_strength_v2(close, vol, high, low)
                if score is not None:
                    self.results[code] = score

                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(stocks)}...")

            print(f"  完成: {len(self.results)} 只有效数据")
            return self.results

        except Exception as e:
            print(f"  ❌ 扫描异常: {e}")
            import traceback; traceback.print_exc()
            return {}

    def _calc_strength_v2(self, close: pd.Series, vol: pd.Series,
                          high: pd.Series, low: pd.Series) -> Optional[dict]:
        """7因子强度计算（v2.0）"""
        try:
            n = len(close)
            latest = close.iloc[-1]

            # --- 因子1: 价格位置（多窗口加权）---
            def _price_pos(series, window):
                if len(series) < window:
                    return 50
                w = series.iloc[-window:]
                if w.max() == w.min():
                    return 50
                return (latest - w.min()) / (w.max() - w.min()) * 100

            pos_30 = _price_pos(close, CONFIG["windows"]["short"])
            pos_90 = _price_pos(close, CONFIG["windows"]["medium"])
            pos_250 = _price_pos(close, CONFIG["windows"]["long"]) if n >= 250 else pos_90
            price_pos = (pos_30 * CONFIG["weights"]["short"] +
                         pos_90 * CONFIG["weights"]["medium"] +
                         pos_250 * CONFIG["weights"]["long"])

            # --- 因子2: 趋势强度（均线排列）---
            ma5 = close.iloc[-5:].mean()
            ma10 = close.iloc[-10:].mean() if n >= 10 else ma5
            ma20 = close.iloc[-20:].mean() if n >= 20 else ma10
            ma60 = close.iloc[-60:].mean() if n >= 60 else ma20

            trend_score = 0
            if latest > ma5 > ma10 > ma20 > ma60:
                trend_score = 25
            elif latest > ma5 > ma10 > ma20:
                trend_score = 20
            elif latest > ma5 > ma10:
                trend_score = 15
            elif latest > ma5:
                trend_score = 10
            elif latest > ma20:
                trend_score = 5
            else:
                trend_score = 0

            # --- 因子3: 动量得分 ---
            ret_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if n >= 6 else 0
            ret_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if n >= 11 else 0
            ret_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if n >= 21 else 0
            momentum_raw = ret_5 * 1.0 + ret_10 * 0.5 + ret_20 * 0.3
            momentum_score = min(max(momentum_raw, 0), 20)

            # --- 因子4: 量价配合 ---
            vol_score = 5
            if len(vol) >= 20:
                vol_5 = vol.iloc[-5:].mean()
                vol_20 = vol.iloc[-20:].mean()
                vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
                if vol_ratio > 1.5 and ret_5 > 0:
                    vol_score = 15
                elif vol_ratio > 1.2 and ret_5 > 0:
                    vol_score = 12
                elif vol_ratio > 1.0 and ret_5 > 0:
                    vol_score = 8
                elif vol_ratio < 0.7 and ret_5 < 0:
                    vol_score = 3

            # --- 因子5: 涨停检测 ---
            limit_up_count = self._count_limit_ups(close, high, low, n)
            if limit_up_count >= 4:
                limit_up_score = 10
            elif limit_up_count >= 3:
                limit_up_score = 8
            elif limit_up_count >= 2:
                limit_up_score = 6
            elif limit_up_count >= 1:
                limit_up_score = 3
            else:
                limit_up_score = 0

            # --- 因子6: 波动率ATR ---
            atr = self._calc_atr(close, high, low, 14)
            atr_score = 5
            if atr is not None and latest > 0:
                atr_pct = atr / latest * 100
                if 2 < atr_pct < 6:
                    atr_score = 10
                elif 1.5 < atr_pct < 8:
                    atr_score = 7
                elif atr_pct > 10:
                    atr_score = 2  # 妖股扣分

            # --- 因子7: 偏离均线衰减 ---
            deviation = (latest / ma60 - 1) * 100 if ma60 > 0 else 0
            deviation_decay = 1.0
            if deviation > 50:
                deviation_decay = max(0.3, 1 - (deviation - 50) / 100)

            # --- 高位钝化处理 ---
            if price_pos > CONFIG["decay_pct_threshold"] * 100:
                decay = max(CONFIG["decay_min_factor"],
                           1 - (price_pos / 100 - CONFIG["decay_pct_threshold"]) * 3)
                price_pos *= decay
                trend_score *= decay
                momentum_score *= decay

            # 阶段涨幅过大强制限制
            stage_return = 0
            if n >= CONFIG["windows"]["medium"]:
                start_p = close.iloc[-CONFIG["windows"]["medium"]]
                stage_return = (latest / start_p - 1)
            if stage_return > CONFIG["stage_return_limit"]:
                price_pos = min(price_pos, CONFIG["position_cap_after_decay"])

            # --- ADX判断市场状态 ---
            adx = self._calc_adx(close, high, low, 14)
            is_trending = adx is not None and adx > 25

            # --- 动态权重 ---
            if is_trending:
                weights = CONFIG["factor_weights_trend"]
            else:
                weights = CONFIG["factor_weights_range"]

            # 综合得分：按权重分配100分
            # 价格位置归一化到0-1
            price_pos_norm = price_pos / 100
            trend_norm = trend_score / 25
            momentum_norm = momentum_score / 20
            vol_norm = vol_score / 15
            limit_up_norm = limit_up_score / 10
            atr_norm = atr_score / 10

            total = (
                price_pos_norm * weights["price_pos"] * 100 +
                trend_norm * weights["trend"] * 100 +
                momentum_norm * weights["momentum"] * 100 +
                vol_norm * weights["vol_price"] * 100 +
                limit_up_norm * weights["limit_up"] * 100 +
                atr_norm * weights["atr"] * 100
            ) * deviation_decay

            # 连板扣分
            streak_penalty = self._calc_streak_penalty(close, high, low, n)
            total = min(100, max(0, total + streak_penalty))

            return {
                "close": round(latest, 2),
                "strength": round(total, 1),
                "price_pct": round(price_pos, 1),
                "ret_5d": round(ret_5, 1),
                "ret_20d": round(ret_20, 1),
                "trend": "多头排列" if trend_score >= 20 else ("偏多" if trend_score >= 10 else "偏弱"),
                "limit_up_days": limit_up_count,
                "is_trending": is_trending,
                "adx": round(adx, 1) if adx else 0,
                "deviation": round(deviation, 1),
            }
        except Exception:
            return None

    def _count_limit_ups(self, close, high, low, n):
        """精确涨停检测（>=9.5%且最高价接近收盘价）"""
        count = 0
        lookback = min(n, CONFIG["windows"]["medium"])
        for i in range(max(1, n - lookback), n):
            try:
                prev_close = close.iloc[i - 1]
                today_high = high.iloc[i] if i < len(high) else close.iloc[i]
                today_close = close.iloc[i]
                chg = (today_close - prev_close) / prev_close * 100
                if chg >= 9.5 and today_high >= today_close * 0.98:
                    count += 1
            except Exception:
                continue
        return count

    def _calc_atr(self, close, high, low, period=14):
        """计算ATR"""
        try:
            if len(close) < period + 1:
                return None
            tr_list = []
            for i in range(max(1, len(close) - period), len(close)):
                h = high.iloc[i] if i < len(high) else close.iloc[i]
                l = low.iloc[i] if i < len(low) else close.iloc[i]
                prev_c = close.iloc[i - 1]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                tr_list.append(tr)
            return np.mean(tr_list) if tr_list else None
        except Exception:
            return None

    def _calc_adx(self, close, high, low, period=14):
        """简化ADX计算"""
        try:
            if len(close) < period * 2:
                return None
            dm_plus = []
            dm_minus = []
            tr_list = []
            for i in range(max(1, len(close) - period * 2), len(close)):
                h = high.iloc[i] if i < len(high) else close.iloc[i]
                l = low.iloc[i] if i < len(low) else close.iloc[i]
                prev_h = high.iloc[i - 1] if i - 1 < len(high) else close.iloc[i - 1]
                prev_l = low.iloc[i - 1] if i - 1 < len(low) else close.iloc[i - 1]
                prev_c = close.iloc[i - 1]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                tr_list.append(tr)
                up = h - prev_h
                down = prev_l - l
                dm_plus.append(up if up > down and up > 0 else 0)
                dm_minus.append(down if down > up and down > 0 else 0)
            if not tr_list:
                return None
            atr_val = np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)
            if atr_val == 0:
                return None
            di_plus = np.mean(dm_plus[-period:]) / atr_val * 100
            di_minus = np.mean(dm_minus[-period:]) / atr_val * 100
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
            return dx
        except Exception:
            return None

    def _calc_streak_penalty(self, close, high, low, n):
        """连板扣分：连续涨停天数越多扣分越多"""
        consecutive = 0
        for i in range(n - 1, max(0, n - 10), -1):
            try:
                prev_c = close.iloc[i - 1]
                today_c = close.iloc[i]
                chg = (today_c - prev_c) / prev_c * 100
                if chg >= 9.5:
                    consecutive += 1
                else:
                    break
            except Exception:
                break
        if consecutive >= 4:
            return -8
        elif consecutive >= 3:
            return -5
        elif consecutive >= 2:
            return -2
        return 0

    def get_top_stocks(self, top_n: int = 50) -> List[Tuple[str, dict]]:
        sorted_stocks = sorted(self.results.items(), key=lambda x: x[1].get("strength", 0), reverse=True)
        return sorted_stocks[:top_n]


# ============================================================
# 第二层：板块聚合 v2.0（精确轮动检测）
# ============================================================
class SectorAggregatorV2:
    """个股强度 → 板块强度 聚合器（含精确轮动检测）"""

    def __init__(self, scanner: StockStrengthScannerV2):
        self.scanner = scanner
        self.sector_stocks = {}
        self.sector_strength = {}
        self.sector_rotation = {}

    def load_sectors(self):
        try:
            if not hasattr(SectorAggregatorV2, "_tq_init"):
                tq.initialize("AGG_V2")
                SectorAggregatorV2._tq_init = True

            all_sectors = tq.get_sector_list(1)
            print(f"  加载 {len(all_sectors)} 个概念板块...")

            for i, sector in enumerate(all_sectors):
                code = sector.get("Code", "")
                name = sector.get("Name", "")
                try:
                    stocks = tq.get_stock_list_in_sector(code, 0, 0)
                    if stocks and len(stocks) >= CONFIG["min_sector_stocks"]:
                        self.sector_stocks[code] = {
                            "name": name, "stocks": stocks, "n_stocks": len(stocks),
                        }
                except Exception:
                    continue
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(all_sectors)}...")

            print(f"  有效板块: {len(self.sector_stocks)} 个")
            return self.sector_stocks
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            return {}

    def aggregate(self) -> Dict:
        if not self.sector_stocks:
            self.load_sectors()

        stock_scores = self.scanner.results
        if not stock_scores:
            print("  ❌ 无个股数据")
            return {}

        print(f"\n  聚合 {len(self.sector_stocks)} 个板块...")
        for sector_code, info in self.sector_stocks.items():
            sector_stocks = info["stocks"]
            scores = []
            strong_count = 0
            limit_up_stocks = {}

            for stock in sector_stocks:
                if stock in stock_scores:
                    s = stock_scores[stock]
                    scores.append(s["strength"])
                    if s["strength"] >= CONFIG["strong_threshold"]:
                        strong_count += 1
                    if s.get("limit_up_days", 0) > 0:
                        limit_up_stocks[stock] = s.get("limit_up_days", 0)

            if len(scores) < CONFIG["min_sector_stocks"]:
                continue

            avg_strength = np.mean(scores)
            strong_ratio = strong_count / len(scores)

            self.sector_strength[sector_code] = {
                "name": info["name"],
                "n_stocks": len(scores),
                "avg_strength": round(avg_strength, 1),
                "strong_ratio": round(strong_ratio, 3),
                "strong_count": strong_count,
                "max_strength": round(max(scores), 1),
                "std_strength": round(np.std(scores), 1),
            }

            # 精确轮动检测
            if len(limit_up_stocks) >= 2:
                rotation = self._detect_rotation(sector_code, info, limit_up_stocks)
                if rotation["quality_score"] >= CONFIG["rotation_quality_threshold"]:
                    self.sector_rotation[sector_code] = rotation

        print(f"  有效板块: {len(self.sector_strength)} 个")
        print(f"  真轮动板块: {len(self.sector_rotation)} 个")
        return self.sector_strength

    def _detect_rotation(self, code: str, info: dict, limit_up_stocks: dict) -> dict:
        """精确轮动检测：轮动度 + 质量评分"""
        n_limit_stocks = len(limit_up_stocks)
        total_limit_ups = sum(limit_up_stocks.values())

        # 轮动度 = 涨停分散程度（多只股票各有1-2次涨停 > 少数股票多次涨停）
        if total_limit_ups > 0:
            avg_per_stock = total_limit_ups / n_limit_stocks
            if avg_per_stock <= 2.5:
                rotation_degree = min(1.0, n_limit_stocks / 8)  # 8只以上满分
            else:
                rotation_degree = min(1.0, n_limit_stocks / 8) * 0.5  # 集中涨停扣分
        else:
            rotation_degree = 0

        # 质量评分 = 轮动度 × 扩散率 × 强度配合
        strong_ratio = self.sector_strength.get(code, {}).get("strong_ratio", 0)
        diffusion = min(1.0, n_limit_stocks / info["n_stocks"] * 3)  # 涨停占比
        strength_factor = min(1.0, strong_ratio / 0.3)  # 强势密度归一化

        quality = rotation_degree * 0.5 + diffusion * 0.3 + strength_factor * 0.2

        is_dragon = any(v >= 3 for v in limit_up_stocks.values())  # 有龙头连板

        return {
            "name": info["name"],
            "n_rotating_stocks": n_limit_stocks,
            "total_limit_ups": total_limit_ups,
            "rotation_degree": round(rotation_degree, 2),
            "quality_score": round(quality, 2),
            "has_dragon": is_dragon,
            "is_genuine_rotation": quality >= CONFIG["rotation_quality_threshold"],
            "stocks_detail": dict(sorted(limit_up_stocks.items(), key=lambda x: -x[1])[:5]),
        }

    def get_high_density_sectors(self, threshold: float = None) -> List[dict]:
        if threshold is None:
            threshold = CONFIG["density_threshold"]
        results = []
        for code, info in self.sector_strength.items():
            if info["strong_ratio"] >= threshold:
                rot = self.sector_rotation.get(code, {})
                results.append({
                    "code": code, "name": info["name"],
                    "strong_ratio": info["strong_ratio"],
                    "strong_count": info["strong_count"],
                    "n_stocks": info["n_stocks"],
                    "avg_strength": info["avg_strength"],
                    "has_rotation": code in self.sector_rotation,
                    "rotation_quality": rot.get("quality_score", 0),
                    "rotating_stocks": rot.get("n_rotating_stocks", 0),
                })
        results.sort(key=lambda x: x["strong_ratio"], reverse=True)
        return results

    def get_rotation_sectors(self) -> List[dict]:
        results = []
        for code, info in self.sector_rotation.items():
            strength_info = self.sector_strength.get(code, {})
            results.append({
                "code": code, "name": info["name"],
                "rotation_degree": info["rotation_degree"],
                "quality_score": info["quality_score"],
                "n_rotating_stocks": info["n_rotating_stocks"],
                "total_limit_ups": info["total_limit_ups"],
                "has_dragon": info["has_dragon"],
                "avg_strength": strength_info.get("avg_strength", 0),
                "strong_ratio": strength_info.get("strong_ratio", 0),
            })
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        return results


# ============================================================
# 第三层：主线映射 v2.0（三维信号 + 过滤 + 交易信号）
# ============================================================
class MainlineMapperV2:
    """板块信号 → 主线 映射器（含假信号过滤和交易信号生成）"""

    def __init__(self, aggregator: SectorAggregatorV2):
        self.aggregator = aggregator
        self.mainline_sectors = self._build_mainline_map()

    def _build_mainline_map(self) -> Dict[str, List[str]]:
        try:
            from mainline_fear_greed_history import MAINLINE_TQ_MAP
            return {k: v for k, v in MAINLINE_TQ_MAP.items()}
        except Exception:
            return {}

    def map_to_mainlines(self) -> Dict:
        mainline_signals = {}
        for mainline, tq_codes in self.mainline_sectors.items():
            sector_data = []
            for code in tq_codes:
                if code in self.aggregator.sector_strength:
                    sector_data.append(self.aggregator.sector_strength[code])

            if not sector_data:
                continue

            strengths = [s["avg_strength"] for s in sector_data]
            ratios = [s["strong_ratio"] for s in sector_data]

            has_rotation = any(code in self.aggregator.sector_rotation for code in tq_codes)
            rot_sectors = [self.aggregator.sector_rotation[c]["name"]
                          for c in tq_codes if c in self.aggregator.sector_rotation]

            # 计算强势股密度趋势（取所有关联板块的加权平均）
            weighted_ratio = np.average(ratios, weights=[len(sector_data) - i for i in range(len(sector_data))]) if len(ratios) > 1 else ratios[0]

            mainline_signals[mainline] = {
                "avg_strength": round(np.mean(strengths), 1) if strengths else 0,
                "avg_strong_ratio": round(np.mean(ratios), 3) if ratios else 0,
                "weighted_ratio": round(weighted_ratio, 3),
                "n_sectors": len(tq_codes),
                "has_rotation": has_rotation,
                "rotation_sectors": rot_sectors,
            }

        return mainline_signals

    def apply_filters(self, mainline_data: Dict) -> Dict:
        """多层假信号过滤"""
        filtered = {}
        for mainline, ml in mainline_data.items():
            strong_ratio = ml["avg_strong_ratio"]
            has_rot = ml["has_rotation"]

            # 过滤1：位置分布检查（强密度但无轮动 = 可能假信号）
            if strong_ratio >= 0.3 and not has_rot:
                ml["filter_note"] = "⚠️ 高密度无轮动，需观察"

            # 过滤2：极端强度检查
            if ml["avg_strength"] >= 80 and strong_ratio >= 0.5:
                ml["filter_note"] = "⚠️ 强度极端，可能过热"

            # 过滤3：标准差过大（分化严重）
            if ml.get("n_sectors", 1) <= 1 and strong_ratio >= 0.4:
                ml["filter_note"] = "⚠️ 单板块信号，需多板块确认"

            filtered[mainline] = ml

        return filtered

    def generate_signals(self, mainline_fg: Dict[str, float]) -> Dict:
        """生成三维信号 + 交易信号"""
        mainline_data = self.apply_filters(self.map_to_mainlines())

        signals = {}
        for mainline in mainline_fg:
            fg = mainline_fg[mainline]
            ml = mainline_data.get(mainline, {})
            strong_ratio = ml.get("avg_strong_ratio", 0)
            weighted_ratio = ml.get("weighted_ratio", strong_ratio)
            has_rot = ml.get("has_rotation", False)
            rot_sectors = ml.get("rotation_sectors", [])

            # --- 三维信号判断 ---
            if fg >= 75 and strong_ratio >= 0.5:
                signal = "🔴 极度泡沫，远离！"
                action = "SELL"
            elif fg >= 70 and strong_ratio >= 0.35:
                signal = "🟠 高位过热+强势密集→泡沫风险"
                action = "REDUCE"
            elif fg >= 70 and strong_ratio < 0.2:
                signal = "⚠️ 高位但强势稀疏→龙头独舞，分化"
                action = "HOLD"
            elif fg <= 25 and strong_ratio >= 0.30 and has_rot:
                signal = "🟢 低位+强势启动+轮动→板块反转！"
                action = "STRONG_BUY"
            elif fg <= 35 and strong_ratio >= 0.20:
                signal = "🟡 低位+强势出现→关注反转"
                action = "BUY"
            elif fg <= 30 and strong_ratio < 0.15:
                signal = "⏳ 低位无强势→真正底部，等待"
                action = "WATCH"
            elif has_rot and strong_ratio >= 0.20:
                signal = "🔥 轮动涨停+强势→资金流入，关注"
                action = "BUY"
            elif has_rot and strong_ratio < 0.10:
                signal = "👀 轮动出现但弱→启动前兆"
                action = "WATCH"
            else:
                signal = "➖ 中性"
                action = "HOLD"

            # --- 交易信号（仓位/止损建议）---
            position_pct = 0
            stop_loss = 0
            if action == "STRONG_BUY":
                position_pct = 40
                stop_loss = -5
            elif action == "BUY":
                position_pct = 20
                stop_loss = -3
            elif action == "WATCH":
                position_pct = 5
                stop_loss = -2

            signals[mainline] = {
                "fg": fg,
                "strong_ratio": strong_ratio,
                "weighted_ratio": weighted_ratio,
                "has_rotation": has_rot,
                "rotation_sectors": rot_sectors,
                "signal": signal,
                "action": action,
                "position_pct": position_pct,
                "stop_loss_pct": stop_loss,
                "filter_note": ml.get("filter_note", ""),
            }

        return signals


# ============================================================
# 主流程
# ============================================================
def run_full_scan(top_n: int = 200):
    """运行完整三层扫描 v2.0"""
    print("=" * 80)
    print("  三层扫描系统 v2.0：个股→板块→主线（LLM完善版）")
    print("=" * 80)

    # 第一层
    print("\n[第一层] 全市场个股强度扫描（7因子+动态权重+高位钝化）...")
    scanner = StockStrengthScannerV2()
    scanner.scan_all(top_n=top_n)

    top_stocks = scanner.get_top_stocks(50)
    print(f"\n  📈 强度最高的15只个股:")
    for code, info in top_stocks[:15]:
        trend_icon = "📈" if info["trend"] == "多头排列" else ("↗️" if info["trend"] == "偏多" else "➡️")
        print(f"    {code}: 强度{info['strength']:.0f}  分位{info['price_pct']:.0f}%  "
              f"5日{info['ret_5d']:+.1f}%  涨停{info['limit_up_days']}次  "
              f"ADX{info.get('adx',0):.0f}  {trend_icon}{info['trend']}")

    # 第二层
    print(f"\n[第二层] 个股→板块 聚合（精确轮动检测）...")
    aggregator = SectorAggregatorV2(scanner)
    aggregator.aggregate()

    high_density = aggregator.get_high_density_sectors()
    print(f"\n  🔥 强势股密度高的板块 (top 15):")
    for i, s in enumerate(high_density[:15], 1):
        rot = f" 🔄{s['rotating_stocks']}只轮动(Q={s['rotation_quality']:.1f})" if s["has_rotation"] else ""
        print(f"    {i:2}. {s['name']:<12} 密度{s['strong_ratio']:.0%} "
              f"({s['strong_count']}/{s['n_stocks']}) 均强{s['avg_strength']:.0f}{rot}")

    rotation_sectors = aggregator.get_rotation_sectors()
    if rotation_sectors:
        print(f"\n  🔄 高质量轮动板块 (Top 10):")
        for s in rotation_sectors[:10]:
            dragon = "🐉" if s["has_dragon"] else "  "
            print(f"    {s['name']:<12} {dragon} 轮动度{s['rotation_degree']:.2f}  "
                  f"质量{s['quality_score']:.2f}  {s['n_rotating_stocks']}只{s['total_limit_ups']}次涨停")

    # 第三层
    print(f"\n[第三层] 板块→主线 三维信号 + 交易信号...")
    mapper = MainlineMapperV2(aggregator)

    REAL_FG = {
        "AI算力/光模块": 72.4, "AI芯片/半导体": 61.0, "存储/国产替代": 62.2,
        "算力/服务器": 53.7, "电力/绿电": 60.5, "新能源/锂电池": 50.6,
        "可控核聚变": 39.8, "机器人/人形机器人": 41.9,
        "智能驾驶/新能源汽车": 58.4, "商业航天/低空经济": 61.4,
        "半导体设备/材料": 53.0, "稀土/资源": 41.7, "军工/国防": 59.0,
        "铜/金属": 65.6, "创新药/医药": 21.0, "华为产业链/信创": 58.3,
        "数字经济/数据要素": 47.3, "消费电子/MR头显": 68.2, "证券/多元金融": 30.9,
    }

    signals = mapper.generate_signals(REAL_FG)

    # 按交易信号优先级排序
    priority_order = {"STRONG_BUY": 0, "BUY": 1, "WATCH": 2, "HOLD": 3, "REDUCE": 4, "SELL": 5}

    print(f"\n  📊 三维信号 + 交易建议:")
    print(f"  {'主线板块':<18} {'恐贪':>5} {'密度':>7} {'轮动':>4} {'仓位':>5} {'止损':>5}  {'信号'}")
    print(f"  {'-'*90}")
    for mainline, sig in sorted(signals.items(),
                                 key=lambda x: priority_order.get(x[1]["action"], 99)):
        rot = "🔄" if sig["has_rotation"] else "  "
        pos = f"{sig['position_pct']}%" if sig["position_pct"] > 0 else "-"
        sl = f"{sig['stop_loss_pct']}%" if sig["stop_loss_pct"] < 0 else "-"
        print(f"  {mainline:<18} {sig['fg']:5.0f} {sig['strong_ratio']:6.0%}  {rot}   {pos:>5} {sl:>5}  {sig['signal']}")

    # 信号汇总
    print(f"\n  📋 交易信号汇总:")
    buy_signals = [(m, s) for m, s in signals.items() if s["action"] in ("STRONG_BUY", "BUY")]
    watch_signals = [(m, s) for m, s in signals.items() if s["action"] == "WATCH"]
    sell_signals = [(m, s) for m, s in signals.items() if s["action"] in ("REDUCE", "SELL")]

    if buy_signals:
        print(f"  🟢 买入信号 ({len(buy_signals)}个):")
        for m, s in buy_signals:
            print(f"      {m}: 仓位{s['position_pct']}%, 止损{s['stop_loss_pct']}%, 恐贪{s['fg']:.0f}")
    if watch_signals:
        print(f"  🟡 观察信号 ({len(watch_signals)}个):")
        for m, s in watch_signals:
            print(f"      {m}: 恐贪{s['fg']:.0f}, 密度{s['strong_ratio']:.0%}")
    if sell_signals:
        print(f"  🔴 减仓/卖出 ({len(sell_signals)}个):")
        for m, s in sell_signals:
            print(f"      {m}: {s['signal']}")

    # 地产相关板块特别关注
    real_estate_related = [s for s in high_density if any(kw in s["name"] for kw in ["地产", "房地产", "建材", "家居", "银行"])]
    if real_estate_related:
        print(f"\n  🏠 地产相关板块:")
        for s in real_estate_related[:5]:
            rot = " 🔄" if s["has_rotation"] else ""
            print(f"      {s['name']}: 密度{s['strong_ratio']:.0%}, 均强{s['avg_strength']:.0f}{rot}")

    return {
        "scanner": scanner,
        "aggregator": aggregator,
        "mapper": mapper,
        "signals": signals,
    }


if __name__ == "__main__":
    result = run_full_scan(top_n=200)
    try:
        tq.close()
    except Exception:
        pass