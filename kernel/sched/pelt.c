// SPDX-License-Identifier: GPL-2.0
/*
 * Per Entity Load Tracking
 *
 *  Copyright (C) 2007 Red Hat, Inc., Ingo Molnar <mingo@redhat.com>
 *
 *  Interactivity improvements by Mike Galbraith
 *  (C) 2007 Mike Galbraith <efault@gmx.de>
 *
 *  Various enhancements by Dmitry Adamushko.
 *  (C) 2007 Dmitry Adamushko <dmitry.adamushko@gmail.com>
 *
 *  Group scheduling enhancements by Srivatsa Vaddagiri
 *  Copyright IBM Corporation, 2007
 *  Author: Srivatsa Vaddagiri <vatsa@linux.vnet.ibm.com>
 *
 *  Scaled math optimizations by Thomas Gleixner
 *  Copyright (C) 2007, Thomas Gleixner <tglx@linutronix.de>
 *
 *  Adaptive scheduling granularity, math enhancements by Peter Zijlstra
 *  Copyright (C) 2007 Red Hat, Inc., Peter Zijlstra
 *
 *  Move PELT related code from fair.c into this pelt.c file
 *  Author: Vincent Guittot <vincent.guittot@linaro.org>
 */

/* 参考文章：
 * https://zhuanlan.zhihu.com/p/158185705
 */

#include <linux/sched.h>
#include "sched.h"
#include "pelt.h"

#include <trace/events/sched.h>

/*
 * Approximate:
 *   val * y^n,    where y^32 ~= 0.5 (~1 scheduling period)
 * decay_load()函数用于计算val*(y^n)的值
 */
static u64 decay_load(u64 val, u64 n)
{
	unsigned int local_n;
	/* LOAD_AVG_PERIOD=32，即32*63（～2016ms）前的负载对当前的贡献忽略不计 */
	if (unlikely(n > LOAD_AVG_PERIOD * 63))
		return 0;

	/* after bounds checking we can collapse to 32-bit */
	local_n = n;

	/*
	 * As y^PERIOD = 1/2, we can combine
	 * 	  y^n = y^((n/PERIOD)*PERIOD + n%PERIOD)
	 * 	  y^n = (y^PERIOD)^(n/PERIOD) * y^(n%PERIOD)
	 *    y^n = 1/2^(n/PERIOD) * y^(n%PERIOD)
	 * With a look-up table which covers y^n (n<PERIOD)
	 * 因为y^32=0.5，所以y^n可以变为1/2^(n/32) * y^(n%32)
	 * To achieve constant time decay_load.
	 */
	if (unlikely(local_n >= LOAD_AVG_PERIOD)) {
		/* val = val * (1/2)^(local_n/32) */
		val >>= local_n / LOAD_AVG_PERIOD;
		/* local_n = local_n%32 */
		local_n %= LOAD_AVG_PERIOD;
	}
	/* 	  将y^(n%32)的一共32中32种取值，提前计算好再乘以2^32（提高计算精度），
     *    结果记录在runnable_avg_yN_inv数组中，这样mul_u64_u32_shr函数计算的
     *    (val*runnable_avg_yN_inv[local_n])/2^32即为val*y^n的值，
     *    此时decay_load()函数的时间复杂度从原始版本的O(n)变为了O(1)
     */
	val = mul_u64_u32_shr(val, runnable_avg_yN_inv[local_n], 32); //shr:右移32bit
	return val;
}

static u32 __accumulate_pelt_segments(u64 periods, u32 d1, u32 d3)
{
	u32 c1, c2, c3 = d3; /* y^0 == 1 */

	/*
	 * c1 = d1 y^p
	 */
	c1 = decay_load((u64)d1, periods);

	/*
	 *            p-1
	 * c2 = 1024 \Sum y^n
	 *            n=1
	 *
	 *              inf        inf
	 *    = 1024 ( \Sum y^n - \Sum y^n - y^0 )   对第二项提取公因子y^p
	 *              n=0        n=p
	 * 
	 *              inf              inf                                        inf
	 *    = 1024 ( \Sum y^n - y^p * \Sum y^n - y^0 )   既然 LOAD_AVG_MAX = 1024 \Sum y^n
	 *              n=0              n=0                                        n=0
	 * 
	 *    = LOAD_AVG_MAX - LOAD_AVG_MAX * y^n - 1024
	 * 
	 *    = LOAD_AVG_MAX - decay_load(LOAD_AVG_MAX, periods) - 1024
	 * 
	 */
	c2 = LOAD_AVG_MAX - decay_load(LOAD_AVG_MAX, periods) - 1024;

	return c1 + c2 + c3;
}

#define cap_scale(v, s) ((v)*(s) >> SCHED_CAPACITY_SHIFT)

/*
 * accumulate_sum()函数负责计算se|cfs_rq当前的负载贡献，接受的形参如下：
 * delta：距离上一次计算的物理时间间隔，单位us
 * sa：se|cfs_rq对应的struct sched_avg成员
 * 
 * Accumulate the three separate parts of the sum; d1 the remainder
 * of the last (incomplete) period, d2 the span of full periods and d3
 * the remainder of the (incomplete) current period.
 *
 *           d1          d2           d3
 *           ^           ^            ^
 *           |           |            |
 *         |<->|<----------------->|<--->|
 * ... |---x---|------| ... |------|-----x (now)
 *
 *                           p-1
 * u' = (u + d1) y^p + 1024 \Sum y^n + d3 y^0
 *                           n=1
 *
 *    = u y^p +					(Step 1)
 *
 *                     p-1
 *      d1 y^p + 1024 \Sum y^n + d3 y^0		(Step 2)
 *                     n=1
 * 
 * Step1是对原来的load/runnable_load/util_sum进行衰减，每次更新都必须进行的。
 * Step2只是计算本delta的时间贡献值，能否贡献到loadrunnable_load/util_sum里，还要看这段时间里se|cfs_rq的load/runnable/running值。
 */
static __always_inline u32
accumulate_sum(u64 delta, struct sched_avg *sa,
	       unsigned long load, unsigned long runnable, int running)
{
	u32 contrib = (u32)delta; /* p == 0 -> delta < 1024 */
	u64 periods;

	delta += sa->period_contrib;
	periods = delta / 1024; /* A period is 1024us (~1ms) */

	/*
	 * Step 1: decay old *_sum if we crossed period boundaries.
	 */
	if (periods) {
		/* 无论本delta期间的负载贡献是否被统计，都需要对上一次计算的负载贡献值进行衰减 */
		sa->load_sum = decay_load(sa->load_sum, periods);
		sa->runnable_load_sum =
			decay_load(sa->runnable_load_sum, periods);
		sa->util_sum = decay_load((u64)(sa->util_sum), periods);

		/*
		 * Step 2
		 */
		delta %= 1024; //计算d3，即本次update时最后未满1024us的部分。
		contrib = __accumulate_pelt_segments(periods, //contrib仅仅是获取的时间贡献。
				1024 - sa->period_contrib, delta);
	}
	sa->period_contrib = delta; /* 更新sa->period_contrib */

	/* 根据load、runnabe、running等条件，决定是否统计本delta的负载贡献。
	 * 对于sched_entity，参考函数__update_load_avg_se()
	 *		load = !!se->on_rq, runnable = !!se->on_rq, running = cfs_rq->curr == se
	 *		除util_sum外，load|runnable_load_sum只是时间贡献值的总和。
	 * 对于cfs_rq
	 */
	if (load)
		sa->load_sum += load * contrib;
	if (runnable)
		sa->runnable_load_sum += runnable * contrib;
	if (running)
		sa->util_sum += contrib << SCHED_CAPACITY_SHIFT;

	return periods;
}

/*
 * 为了实现sched entity级别的负载跟踪，pelt将物理时间划分成了1ms（实际为1024us）的序列，
 * 在每一个1024us的周期中，sched entity对系统负载的贡献可以根据该entity处于runnable状态
 * （包含在cpu上running和在cfs_rq上waiting的状态）的时间进行计算；假设该周期内，
 * 某entity处于runnable状态的时间为x，那么其对系统负载的贡献为（x/1024）；
 * pelt还会累积过去周期的负载贡献，但会根据时间乘以相应的衰减系数y，
 * 假设Li表示某entity在周期Pi中对系统负载的贡献，那么该entity对系统负载的总贡献为：
 * L = L0 + L1*y + L2*y^2 + L3*y^3+... (y^32=0.5)
 * 
 * We can represent the historical contribution to runnable average as the
 * coefficients of a geometric series.  To do this we sub-divide our runnable
 * history into segments of approximately 1ms (1024us); label the segment that
 * occurred N-ms ago p_N, with p_0 corresponding to the current period, e.g.
 *
 * [<- 1024us ->|<- 1024us ->|<- 1024us ->| ...
 *      p0            p1           p2
 *     (now)       (~1ms ago)  (~2ms ago)
 *
 * Let u_i denote the fraction of p_i that the entity was runnable.
 *
 * We then designate the fractions u_i as our co-efficients, yielding the
 * following representation of historical load:
 *   u_0 + u_1*y + u_2*y^2 + u_3*y^3 + ...
 *
 * We choose y based on the with of a reasonably scheduling period, fixing:
 *   y^32 = 0.5
 *
 * This means that the contribution to load ~32ms ago (u_32) will be weighted
 * approximately half as much as the contribution to load within the last ms
 * (u_0).
 *
 * When a period "rolls over" and we have new u_0`, multiplying the previous
 * sum again by y is sufficient to update:
 *   load_avg = u_0` + y*(u_0 + u_1*y + u_2*y^2 + ... )
 *            = u_0 + u_1*y + u_2*y^2 + ... [re-labeling u_i --> u_{i+1}]
 */
static __always_inline int
___update_load_sum(u64 now, struct sched_avg *sa,
		  unsigned long load, unsigned long runnable, int running)
{
	u64 delta;

	delta = now - sa->last_update_time; //delta单位为ns
	/*
	 * This should only happen when time goes backwards, which it
	 * unfortunately does during sched clock init when we swap over to TSC.
	 */
	if ((s64)delta < 0) {
		sa->last_update_time = now;
		return 0;
	}

	/*
	 * Use 1024ns as the unit of measurement since it's a reasonable
	 * approximation of 1us and fast to compute.
	 */
	delta >>= 10; //delta由ns近似转换为us
	if (!delta) //不足1us没必要进行衰减等计算，直接返回
		return 0;

	sa->last_update_time += delta << 10; //更新last_update_time，单位ns，所以需要delta << 10.

	/*
	 * running is a subset of runnable (weight) so running can't be set if
	 * runnable is clear. But there are some corner cases where the current
	 * se has been already dequeued but cfs_rq->curr still points to it.
	 * This means that weight will be 0 but not running for a sched_entity
	 * but also for a cfs_rq if the latter becomes idle. As an example,
	 * this happens during idle_balance() which calls
	 * update_blocked_averages() cfs_rq为空时其load也为0.
	 */
	if (!load)
		runnable = running = 0;

	/*
	 * Now we know we crossed measurement unit boundaries. The *_avg
	 * accrues by two steps:
	 *
	 * Step 1: accumulate *_sum since last_update_time. If we haven't
	 * crossed period boundaries, finish.
	 * 调用accumulate_sum()函数对负载贡献进行计算，并依据更新条件更新{load|runnable|util}_sum值
	 */
	if (!accumulate_sum(delta, sa, load, runnable, running))
		return 0;

	return 1;
}

static __always_inline void
___update_load_avg(struct sched_avg *sa, unsigned long load, unsigned long runnable)
{
	/* 在commit 625ed2bf049d5（sched/cfs: Make util/load_avg more stable）
	 * 中对此对了解释：
	 * LOAD_AVG_MAX*y + 1024(us) = LOAD_AVG_MAX        (1)
	 * max_value = LOAD_AVG_MAX*y + sa->period_contrib (2)
	 * 根据(1)、(2)，精确的负载最大值应为LOAD_AVG_MAX - 1024 + sa->period_contrib
	 */
	u32 divider = LOAD_AVG_MAX - 1024 + sa->period_contrib;

	/*
	 * Step 2: update *_avg.
	 * 更新 {load|runnable|util}_avg:
	 *        sa->load_avg = (load * sa->load_sum) / divider
	 *        sa->runnable_avg = runnable * sa->runnable_load_sum / divider
	 *        sa->util_avg = sa->util_sum / dividers
	 */
	sa->load_avg = div_u64(load * sa->load_sum, divider);
	sa->runnable_load_avg =	div_u64(runnable * sa->runnable_load_sum, divider);
	WRITE_ONCE(sa->util_avg, sa->util_sum / divider);
}

/*
 * sched_entity:
 *
 *   task:
 *     se_runnable() == se_weight()
 *
 *   group: [ see update_cfs_group() ]
 *     se_weight()   = tg->weight * grq->load_avg / tg->load_avg
 *     se_runnable() = se_weight(se) * grq->runnable_load_avg / grq->load_avg
 *
 *   load_sum := runnable_sum
 *   load_avg = se_weight(se) * runnable_avg
 *
 *   runnable_load_sum := runnable_sum
 *   runnable_load_avg = se_runnable(se) * runnable_avg
 *
 * XXX collapse load_sum and runnable_load_sum
 *
 * cfq_rq:
 *
 *   load_sum = \Sum se_weight(se) * se->avg.load_sum
 *   load_avg = \Sum se->avg.load_avg
 *
 *   runnable_load_sum = \Sum se_runnable(se) * se->avg.runnable_load_sum
 *   runnable_load_avg = \Sum se->avg.runable_load_avg
 */

int __update_load_avg_blocked_se(u64 now, struct sched_entity *se)
{
	if (___update_load_sum(now, &se->avg, 0, 0, 0)) {
		/* 对于se的sa->load|runnable_load_sum，在___update_load_sum()仅仅计算出了总的时间贡献值，
		 * 所以要在___update_load_avg()里传入se_weight(se)和se_runnable(se)来计算负载的平均值。
		 */
		___update_load_avg(&se->avg, se_weight(se), se_runnable(se));
		trace_pelt_se_tp(se);
		return 1;
	}

	return 0;
}

int __update_load_avg_se(u64 now, struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	if (___update_load_sum(now, &se->avg, !!se->on_rq, !!se->on_rq,
				cfs_rq->curr == se)) {
		___update_load_avg(&se->avg, se_weight(se), se_runnable(se));
		cfs_se_util_change(&se->avg);
		trace_pelt_se_tp(se);
		return 1;
	}

	return 0;
}

int __update_load_avg_cfs_rq(u64 now, struct cfs_rq *cfs_rq)
{
	if (___update_load_sum(now, &cfs_rq->avg,
				scale_load_down(cfs_rq->load.weight),
				scale_load_down(cfs_rq->runnable_weight),
				cfs_rq->curr != NULL)) {

		___update_load_avg(&cfs_rq->avg, 1, 1);
		trace_pelt_cfs_tp(cfs_rq);
		return 1;
	}

	return 0;
}

/*
 * rt_rq:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_load_sum = load_sum
 *
 *   load_avg and runnable_load_avg are not supported and meaningless.
 *
 */

int update_rt_rq_load_avg(u64 now, struct rq *rq, int running)
{
	if (___update_load_sum(now, &rq->avg_rt,
				running,
				running,
				running)) {

		___update_load_avg(&rq->avg_rt, 1, 1);
		trace_pelt_rt_tp(rq);
		return 1;
	}

	return 0;
}

/*
 * dl_rq:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_load_sum = load_sum
 *
 */

int update_dl_rq_load_avg(u64 now, struct rq *rq, int running)
{
	if (___update_load_sum(now, &rq->avg_dl,
				running,
				running,
				running)) {

		___update_load_avg(&rq->avg_dl, 1, 1);
		trace_pelt_dl_tp(rq);
		return 1;
	}

	return 0;
}

#ifdef CONFIG_HAVE_SCHED_AVG_IRQ
/*
 * irq:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_load_sum = load_sum
 *
 */

int update_irq_load_avg(struct rq *rq, u64 running)
{
	int ret = 0;

	/*
	 * We can't use clock_pelt because irq time is not accounted in
	 * clock_task. Instead we directly scale the running time to
	 * reflect the real amount of computation
	 */
	running = cap_scale(running, arch_scale_freq_capacity(cpu_of(rq)));
	running = cap_scale(running, arch_scale_cpu_capacity(cpu_of(rq)));

	/*
	 * We know the time that has been used by interrupt since last update
	 * but we don't when. Let be pessimistic and assume that interrupt has
	 * happened just before the update. This is not so far from reality
	 * because interrupt will most probably wake up task and trig an update
	 * of rq clock during which the metric is updated.
	 * We start to decay with normal context time and then we add the
	 * interrupt context time.
	 * We can safely remove running from rq->clock because
	 * rq->clock += delta with delta >= running
	 */
	ret = ___update_load_sum(rq->clock - running, &rq->avg_irq,
				0,
				0,
				0);
	ret += ___update_load_sum(rq->clock, &rq->avg_irq,
				1,
				1,
				1);

	if (ret) {
		___update_load_avg(&rq->avg_irq, 1, 1);
		trace_pelt_irq_tp(rq);
	}

	return ret;
}
#endif
