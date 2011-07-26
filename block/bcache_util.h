
#ifndef _BCACHE_UTIL_H
#define _BCACHE_UTIL_H

#include <linux/errno.h>
#include <linux/ctype.h>
#include <linux/kernel.h>
#include <linux/workqueue.h>

#define USHRT_MAX	((uint16_t) ~0)
#define REQ_WRITE	WRITE
#define REQ_SYNC	(1U << BIO_RW_SYNCIO)
#define REQ_UNPLUG	(1U << BIO_RW_UNPLUG)
#define REQ_META	(1U << BIO_RW_META)
extern struct workqueue_struct *system_wq;

#define PAGE_SECTORS		(PAGE_SIZE / 512)

struct closure;

#include <trace/events/bcache.h>

#ifdef EDEBUG

#define atomic_dec_bug(v)	BUG_ON(atomic_dec_return(v) < 0)
#define atomic_inc_bug(v, i)	BUG_ON(atomic_inc_return(v) <= i)

#else /* EDEBUG */

#define atomic_dec_bug(v)	atomic_dec(v)
#define atomic_inc_bug(v, i)	atomic_inc(v)

#endif

#define BITMASK(name, type, field, offset, size)		\
static inline uint64_t name(const type *k)			\
{ return (k->field >> offset) & ~(((uint64_t) ~0) << size); }	\
								\
static inline void SET_##name(type *k, uint64_t v)		\
{								\
	k->field &= ~(~((uint64_t) ~0 << size) << offset);	\
	k->field |= v << offset;				\
}

#define DECLARE_HEAP(type, name)					\
	struct {							\
		size_t size, heap_size;					\
		type *data;						\
	} name

#define DEFINE_HEAP(type, name, s)					\
	struct {							\
		size_t size;						\
		const size_t heap_size;					\
		type data[s];						\
	} name = { .size = 0, .heap_size = s }

#define heap_for_each(c, heap)						\
	for (size_t _i = 0; c = (heap)->data[_i], _i < (heap)->size; _i++)

#define init_heap(h, s, gfp)						\
({									\
	(h)->size = 0;							\
	(h)->heap_size = s;						\
	if ((h)->heap_size * sizeof(*(h)->data) >= KMALLOC_MAX_SIZE)	\
		(h)->data = vmalloc(s * sizeof(*(h)->data));		\
	else if (s > 0)							\
		(h)->data = kmalloc(s * sizeof(*(h)->data), gfp);	\
	(h)->data;							\
})

#define free_heap(h)							\
do {									\
	if ((h)->heap_size * sizeof(*(h)->data) >= KMALLOC_MAX_SIZE)	\
		vfree((h)->data);					\
	else								\
		kfree((h)->data);					\
} while (0)

#define heap_swap(h, i, j, member)					\
do {									\
	swap((h)->data[i], (h)->data[j]);				\
	(h)->data[i]->member = i;					\
	(h)->data[j]->member = j;					\
} while (0)

#define heap_sift(h, i, member, cmp)					\
do {									\
	long _r, _j = i;						\
									\
	for (; _j * 2 + 1 < (h)->size; _j = _r) {			\
		_r = _j * 2 + 1;					\
		if (_r + 1 < (h)->size &&				\
		    cmp((h)->data[_r], (h)->data[_r + 1]))		\
			_r++;						\
									\
		if (cmp((h)->data[_r], (h)->data[_j]))			\
			break;						\
		heap_swap(h, _r, _j, member);				\
	}								\
} while (0)

#define heap_sift_down(h, i, member, cmp)				\
do {									\
	while (i) {							\
		long p = (i - 1) / 2;					\
		if (cmp((h)->data[i], (h)->data[p]))			\
			break;						\
		heap_swap(h, i, p, member);				\
		i = p;							\
	}								\
} while (0)

#define heap_add(h, d, member, cmp)					\
do {									\
	long _i = (d)->member;						\
									\
	if (_i == -1) {							\
		_i = (h)->size++;					\
		(h)->data[_i]  = d;					\
		(d)->member = _i;					\
	}								\
									\
	heap_sift_down(h, _i, member, cmp);				\
	heap_sift(h, _i, member, cmp);					\
} while (0)

#define heap_pop(h, member, cmp)					\
({									\
	typeof((h)->data[0]) _r = (h)->data[0];				\
									\
	if ((h)->size) {						\
		(h)->size--;						\
		heap_swap(h, 0, (h)->size, member);			\
		heap_sift(h, 0, member, cmp);				\
		(h)->data[(h)->size] = NULL;				\
		_r->member = -1;					\
	} else								\
		_r = NULL;						\
	_r;								\
})

#define heap_remove(h, d, member, cmp)					\
do {									\
	long _i = (d)->member;						\
									\
	if (_i == -1)							\
		break;							\
									\
	if (_i != --((h)->size)) {					\
		heap_swap(h, _i, (h)->size, member);			\
		heap_sift_down(h, _i, member, cmp);			\
		heap_sift(h, _i, member, cmp);				\
	}								\
									\
	(h)->data[(h)->size] = NULL;					\
	(d)->member = -1;						\
} while (0)

#define heap_peek(h)	((h)->size ? (h)->data[0] : NULL)

#define DECLARE_FIFO(type, name)					\
	struct {							\
		size_t front, back, size;				\
		type *data;						\
	} name

#define fifo_for_each(c, fifo)						\
	for (size_t _i = (fifo)->front;					\
	     c = (fifo)->data[_i], _i != (fifo)->back;			\
	     _i = (_i + 1) & (fifo)->size)

#define init_fifo(f, s, gfp)						\
({									\
	BUG_ON(!s);							\
	(f)->front = (f)->back = 0;					\
	(f)->size = roundup_pow_of_two(s);				\
	(f)->data = ((f)->size * sizeof(*(f)->data) >= KMALLOC_MAX_SIZE)\
		? vmalloc((f)->size-- * sizeof(*(f)->data))		\
		: kmalloc((f)->size-- * sizeof(*(f)->data), gfp);	\
	(f)->data;							\
})

#define free_fifo(fifo)							\
do {									\
	if ((fifo)->size * sizeof(*(fifo)->data) >= KMALLOC_MAX_SIZE)	\
		vfree((fifo)->data);					\
	else								\
		kfree((fifo)->data);					\
	(fifo)->data = NULL;						\
} while (0)

#define fifo_used(fifo)		(((fifo)->back - (fifo)->front) & (fifo)->size)
#define fifo_free(fifo)		((fifo)->size - fifo_used(fifo))
#define fifo_full(fifo)		(fifo_free(fifo) == 0)
#define fifo_empty(fifo)	((fifo)->front == (fifo)->back)

#define fifo_front(fifo)	((fifo)->data[(fifo)->front])
#define fifo_back(fifo)							\
	((fifo)->data[((fifo)->back - 1) & (fifo)->size])

#define fifo_idx(fifo, p)						\
	((((p) - (fifo)->data) - (fifo)->front) & (fifo)->size)

#define fifo_push(fifo, i)						\
({									\
	bool _r = !fifo_full(fifo);					\
	if (_r) {							\
		(fifo)->data[(fifo)->back++] = i;			\
		(fifo)->back &= (fifo)->size;				\
	}								\
	_r;								\
})

#define fifo_pop(fifo, i)						\
({									\
	bool _r = !fifo_empty(fifo);					\
	if (_r) {							\
		i = (fifo)->data[(fifo)->front++];			\
		(fifo)->front &= (fifo)->size;				\
	}								\
	_r;								\
})

#define fifo_swap(l, r)							\
do {									\
	swap((l)->front, (r)->front);					\
	swap((l)->back, (r)->back);					\
	swap((l)->size, (r)->size);					\
	swap((l)->data, (r)->data);					\
} while (0)

#define fifo_move(dest, src)						\
do {									\
	typeof(*((dest)->data)) _t;					\
	while (!fifo_full(dest) &&					\
	       fifo_pop(src, _t))					\
		fifo_push(dest, _t);					\
} while (0)

/*
 * These are subject to the infamous aba problem...
 */

#define lockless_list_push(new, list, member)				\
	do {								\
		(new)->member = list;					\
	} while (cmpxchg(&(list), (new)->member, new) != (new)->member)	\

#define lockless_list_pop(list, member) ({				\
	typeof(list) _r;						\
	do {								\
		_r = list;						\
	} while (_r && cmpxchg(&(list), _r, _r->member) != _r);		\
	_r; })

#define ANYSINT_MAX(t)						\
	((((t) 1 << (sizeof(t) * 8 - 2)) - (t) 1) * (t) 2 + (t) 1)

int strtol_h(const char *, long *);
int strtoll_h(const char *, long long *);
int strtoul_h(const char *, unsigned long *);
int strtoull_h(const char *, unsigned long long *);

#define strtoi_h(cp, res)						\
	(__builtin_types_compatible_p(typeof(*res), long)		\
	? strtol_h(cp, (void *) res)					\
	: __builtin_types_compatible_p(typeof(*res), long long)		\
	? strtoll_h(cp, (void *) res)					\
	: __builtin_types_compatible_p(typeof(*res), unsigned long)	\
	? strtoul_h(cp, (void *) res)					\
	: __builtin_types_compatible_p(typeof(*res), unsigned long long)\
	? strtoull_h(cp, (void *) res) : -EINVAL)

ssize_t hprint(char *buf, int64_t v);
bool is_zero(const char *p, size_t n);
int parse_uuid(const char *s, char *uuid);

#define RB_INSERT(root, new, member, cmp)				\
({									\
	__label__ dup;							\
	struct rb_node **n = &(root)->rb_node, *parent = NULL;		\
	typeof(new) this;						\
	int res, ret = -1;						\
									\
	while (*n) {							\
		parent = *n;						\
		this = container_of(*n, typeof(*(new)), member);	\
		res = cmp(new, this);					\
		if (!res)						\
			goto dup;					\
		n = res < 0						\
			? &(*n)->rb_left				\
			: &(*n)->rb_right;				\
	}								\
									\
	rb_link_node(&(new)->member, parent, n);			\
	rb_insert_color(&(new)->member, root);				\
	ret = 0;							\
dup:									\
	ret;								\
})

#define RB_SEARCH(root, search, member, cmp)				\
({									\
	struct rb_node *n = (root)->rb_node;				\
	typeof(&(search)) this, ret = NULL;				\
	int res;							\
									\
	while (n) {							\
		this = container_of(n, typeof(search), member);		\
		res = cmp(&(search), this);				\
		if (!res) {						\
			ret = this;					\
			break;						\
		}							\
		n = res < 0						\
			? n->rb_left					\
			: n->rb_right;					\
	}								\
	ret;								\
})

#define RB_GREATER(root, search, member, cmp)				\
({									\
	struct rb_node *n = (root)->rb_node;				\
	typeof(&(search)) this, ret = NULL;				\
	int res;							\
									\
	while (n) {							\
		this = container_of(n, typeof(search), member);		\
		res = cmp(&(search), this);				\
		if (res < 0) {						\
			ret = this;					\
			n = n->rb_left;					\
		} else							\
			n = n->rb_right;				\
	}								\
	ret;								\
})

#define RB_FIRST(root, type, member)					\
	(root ? container_of(rb_first(root), type, member) : NULL)

#define RB_LAST(root, type, member)					\
	(root ? container_of(rb_last(root), type, member) : NULL)

#define RB_PREV(node, type, member)					\
	(rb_prev(node) ? container_of(rb_prev(node), type, member) : NULL)

#define RB_NEXT(node, type, member)					\
	(rb_next(node) ? container_of(rb_next(node), type, member) : NULL)

#define bio_end(bio)	((bio)->bi_sector + bio_sectors(bio))

void bio_reset(struct bio *bio);
void bio_map(struct bio *bio, void *base);
struct bio *bio_split_front(struct bio *, int, gfp_t, struct bio_set *);
int bio_submit_split(struct bio *bio, atomic_t *i, struct bio_set *bs);
unsigned __bio_max_sectors(struct bio *bio, struct block_device *bdev,
			   sector_t sector);

int bio_alloc_pages(struct bio *bio, gfp_t gfp);

static inline unsigned bio_max_sectors(struct bio *bio)
{
	return __bio_max_sectors(bio, bio->bi_bdev, bio->bi_sector);
}

#ifdef DEBUG_LATENCY

#define pr_latency(j, fmt, ...)						\
do {									\
	int _ms = jiffies_to_msecs(jiffies - (j));			\
	if (j && latency_warn_ms && (_ms) > (int) latency_warn_ms)	\
		printk_ratelimited(KERN_DEBUG "bcache: %i ms latency "	\
			"called from %pf for " fmt "\n", _ms,		\
		       __builtin_return_address(0), ##__VA_ARGS__);	\
} while (0)

#define set_wait(f)	((f)->wait_time = jiffies)

#else
#define pr_latency(...) do {} while (0)
#define set_wait(j)	do {} while (0)
#endif

typedef void (closure_fn) (struct closure *);

typedef struct {
	struct closure *head;
} closure_list_t;

struct closure {
	union {
		struct {
			long			_pad;
			struct task_struct	*p;
			struct closure		*next;
			closure_fn		*fn;
		};
		struct work_struct	w;
	};

	struct closure		*parent;

	union {
		struct {
			atomic_t		_pad2;
			atomic_t		remaining;
		};

#define	CLOSURE_BLOCK		0
#define CLOSURE_NOQUEUE		1
#define __CLOSURE_STACK		2
#define	__CLOSURE_WAITING	3
#define	__CLOSURE_SLEEPING	4
		unsigned long		flags;
	};

#ifdef CLOSURE_DEBUG
	struct list_head	all;
	unsigned long		waiting_on;
#endif
#ifdef DEBUG_LATENCY
	unsigned long		wait_time;
#endif
};

void closure_put(struct closure *s, struct workqueue_struct *wq);
void __closure_init(struct closure *c, struct closure *parent, bool onstack);
void closure_run_wait(closure_list_t *list, struct workqueue_struct *wq);
bool closure_wait(closure_list_t *list, struct closure *c);
void closure_sync(struct closure *c);
void __closure_sleep(struct closure *c);

#ifdef CLOSURE_DEBUG
extern struct list_head closures;
extern spinlock_t closure_lock;

static inline void closure_del(struct closure *c)
{
	unsigned long flags;
	spin_lock_irqsave(&closure_lock, flags);
	list_del(&c->all);
	spin_unlock_irqrestore(&closure_lock, flags);
}

#else
static inline void closure_del(struct closure *c) {}
#endif

static inline void closure_init(struct closure *c, struct closure *parent)
{
	__closure_init(c, parent, false);
}

static inline void closure_init_stack(struct closure *c)
{
	__closure_init(c, NULL, true);
	set_bit(CLOSURE_BLOCK, &c->flags);
	set_bit(__CLOSURE_STACK, &c->flags);
}

static inline void closure_get(struct closure *c)
{
	atomic_inc_bug(&c->remaining, 1);
}

#define __closure_wait_on(list, wq, c, condition, block)		\
({									\
	__label__ out;							\
	typeof(condition) ret;						\
	while (!(ret = (condition))) {					\
		if (block)						\
			__closure_sleep(c);				\
		if (!closure_wait(list, c)) {				\
			if (!block)					\
				goto out;				\
			schedule();					\
		}							\
	}								\
	closure_run_wait(list, wq);					\
	if (block) {							\
		__set_current_state(TASK_RUNNING);			\
		clear_bit(__CLOSURE_SLEEPING, &(c)->flags);		\
	}								\
out:	ret;								\
})

#define closure_wait_on(list, wq, c, condition)				\
	__closure_wait_on(list, wq, c, condition,			\
			  test_bit(CLOSURE_BLOCK, &(c)->flags))

#define closure_wait_on_async(list, wq, c, condition)			\
	__closure_wait_on(list, wq, c, condition, false)

#define return_f(_c, _f, ...)						\
do {									\
	BUG_ON(!(_c) || object_is_on_stack(_c));			\
	clear_bit(CLOSURE_BLOCK, &(_c)->flags);				\
	(_c)->fn = _f;							\
	smp_mb__before_atomic_dec();					\
	closure_put(_c, current->bio_list ? delayed : NULL);		\
	return __VA_ARGS__;						\
} while (0)

#define closure_bio_submit(bio, c, bs)					\
	bio_submit_split(bio, &(c)->remaining, bs)

uint64_t crc64(const void *, size_t);

#endif /* _BCACHE_UTIL_H */
