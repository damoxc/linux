/*
 * Copyright (C) 2010 Kent Overstreet <kent.overstreet@gmail.com>
 *
 * Uses a block device as cache for other block devices; optimized for SSDs.
 * All allocation is done in buckets, which should match the erase block size
 * of the device.
 *
 * Buckets containing cached data are kept on a heap sorted by priority;
 * bucket priority is increased on cache hit, and periodically all the buckets
 * on the heap have their priority scaled down. This currently is just used as
 * an LRU but in the future should allow for more intelligent heuristics.
 *
 * Buckets have an 8 bit counter; freeing is accomplished by incrementing the
 * counter. Garbage collection is used to remove stale pointers.
 *
 * Indexing is done via a btree; nodes are not necessarily fully sorted, rather
 * as keys are inserted we only sort the pages that have not yet been written.
 * When garbage collection is run, we resort the entire node.
 *
 * All configuration is done via sysfs; see Documentation/bcache.txt.
 */

#define pr_fmt(fmt) "bcache: %s() " fmt "\n", __func__

#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/console.h>
#include <linux/ctype.h>
#include <linux/debugfs.h>
#include <linux/delay.h>
#include <linux/device.h>
#include <linux/hash.h>
#include <linux/init.h>
#include <linux/kobject.h>
#include <linux/list.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/random.h>
#include <linux/ratelimit.h>
#include <linux/rcupdate.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/string.h>
#include <linux/swap.h>
#include <linux/sysfs.h>
#include <linux/types.h>
#include <linux/workqueue.h>

#include "bcache_util.h"

/*
 * Todo:
 * register_bcache: Return errors out to userspace correctly
 *
 * Writeback: don't undirty key until after a cache flush
 * Create an iterator for key pointers
 *
 * On btree write error, mark bucket such that it won't be freed from the cache
 *
 * Journalling:
 *   Check for bad keys in replay
 *   Propagate barriers
 *   Refcount journal entries in journal_replay
 *
 * Garbage collection:
 *   Finish incremental gc
 *   Track number of gcs, average gc time and sigma, export in sysfs
 *   Trace via blktrace when gc starts and ends
 *   Gc should free old UUIDs, data for invalid UUIDs
 *   Need to wait on all writes to complete
 *
 * Provide a way to list backing device UUIDs we have data cached for, and
 * probably how long it's been since we've seen them, and a way to invalidate
 * dirty data for devices that will never be attached again
 *
 * Keep 1 min/5 min/15 min statistics of how busy a block device has been, so
 * that based on that and how much dirty data we have we can keep writeback
 * from being starved
 *
 * Add a tracepoint or somesuch to watch for writeback starvation
 *
 * When btree depth > 1 and splitting an interior node, we have to make sure
 * alloc_bucket() cannot fail. This should be true but is not completely
 * obvious.
 *
 * Don't keep the full heap around, build a small heap when we need to that
 * doesn't have backpointers
 *
 * Make sure all allocations get charged to the root cgroup
 *
 * bucket_lock shouldn't be in any fastpaths anymore - verify and turn it into
 * a mutex?
 *
 * Plugging?
 *
 * If data write is less than hard sector size of ssd, round up offset in open
 * bucket to the next whole sector
 *
 * Also lookup by cgroup in get_open_bucket()
 *
 * Superblock needs to be fleshed out for multiple cache devices
 *
 * Add a sysfs tunable for the number of writeback IOs in flight
 *
 * Add a sysfs tunable for the number of open data buckets
 *
 * IO tracking: Can we track when one process is doing io on behalf of another?
 * IO tracking: Don't use just an average, weigh more recent stuff higher
 *
 * Test module load/unload
 */

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kent Overstreet <kent.overstreet@gmail.com>");

#define SB_SECTOR		8
#define SB_SIZE			4096
#define MAX_CACHES_PER_SET	8

struct btree;
struct btree_iter;

static const char bcache_magic[] = {
	0xc6, 0x85, 0x73, 0xf6, 0x4e, 0x1a, 0x45, 0xca,
	0x82, 0x65, 0xf5, 0x7f, 0x48, 0xba, 0x6d, 0x81
};

static const char invalid_uuid[] = {
	0xa0, 0x3e, 0xf8, 0xed, 0x3e, 0xe1, 0xb8, 0x78,
	0xc8, 0x50, 0xfc, 0x5e, 0xcb, 0x16, 0xcd, 0x99
};

#define bucket_prio(i)		(((int) (i->prio - c->set->min_prio)) * i->mark)
#define bucket_cmp(i, j)	(bucket_prio(i) >= bucket_prio(j))

struct bucket {
	long		heap;
	atomic_t	pin;
	uint16_t	prio;
	uint8_t		gen;
	uint8_t		disk_gen;
	uint8_t		last_gc; /* Most out of date gen in the btree */
	uint8_t		gc_gen;

#define GC_MARK_DIRTY	-1
#define GC_MARK_BTREE	-2
	short		mark;
};

struct bkey {
	uint64_t	header;
	uint64_t	key;
	uint64_t	ptr[];
};

#define BKEY_PADDED(key)					\
	union { struct bkey key; uint64_t key ## _pad[8]; }

struct bset {
	uint64_t		csum;
	uint64_t		magic;
	uint64_t		seq;
	uint32_t		version;
	uint32_t		keys;

	union {
		struct bkey	start[0];
		uint64_t	d[0];
	};
};

struct prio_set {
	uint64_t		csum;
	uint64_t		magic;
	uint64_t		seq;
	uint32_t		version;
	uint32_t		pad;

	uint64_t		next_bucket;

	struct bucket_disk {
		uint16_t	prio;
		uint8_t		gen;
	} __attribute((packed)) data[];
};

struct discard {
	struct list_head	list;
	struct work_struct	work;
	struct cache		*c;
	long			bucket;

	struct bio		bio;
	struct bio_vec		bv;
};

struct jset {
	uint64_t		csum;
	uint64_t		magic;
	uint64_t		seq;
	uint32_t		version;
	uint32_t		keys;

	uint64_t		last_seq;

	BKEY_PADDED(uuid_bucket);
	BKEY_PADDED(btree_root);
	uint16_t		btree_level;
	uint16_t		pad[3];

	uint64_t		prio_bucket[MAX_CACHES_PER_SET];

	union {
		struct bkey	start[0];
		uint64_t	d[0];
	};
};

struct journal_replay {
	struct list_head	list;
	struct jset		j;
};

struct journal_seq {
	uint64_t	seq;
	sector_t	sector;
};

struct journal_write {
	BKEY_PADDED(key);

	struct jset		*data;
#define JSET_BITS		3

	struct cache_set	*c;
	closure_list_t		wait;
	bool			need_write;
};

struct journal {
	struct work_struct	work;
	spinlock_t		lock;
	/* used when waiting because the journal was full */
	closure_list_t		wait;
	atomic_t		io;

	unsigned		sectors_free;
	uint64_t		seq;
	DECLARE_FIFO(atomic_t, pin);

	struct journal_write	w[2], *cur;
};

#define journal_pin_cmp(c, l, r)				\
	(fifo_idx(&(c)->journal.pin, (l)->journal) >		\
	 fifo_idx(&(c)->journal.pin, (r)->journal))

struct uuid_entry {
	uint8_t		uuid[16];
	uint8_t		label[32];
	uint32_t	first_reg;
	uint32_t	last_reg;
	uint32_t	invalidated;
	uint32_t	pad;
};

#define SB_LABEL_SIZE		32
#define SB_JOURNAL_BUCKETS	256

struct cache_sb {
	uint64_t		csum;
	uint64_t		offset;	/* sector where this sb was written */
	uint64_t		version;
#define CACHE_BACKING_DEV	1

	uint8_t			magic[16];

	uint8_t			uuid[16];
	union {
		uint8_t		set_uuid[16];
		uint64_t	set_magic;
	};
	uint8_t			label[SB_LABEL_SIZE];

	uint64_t		flags;
	uint64_t		seq;
	uint64_t		pad[8];

	uint64_t		nbuckets;	/* device size */
	uint16_t		block_size;	/* sectors */
	uint16_t		bucket_size;	/* sectors */

	uint16_t		nr_in_set;
	uint16_t		nr_this_dev;

	uint32_t		last_mount;	/* time_t */

	uint16_t		first_bucket;
	uint16_t		keys;		/* number of journal buckets */
	uint64_t		d[SB_JOURNAL_BUCKETS];	/* journal buckets */
};

BITMASK(CACHE_SYNC,	struct cache_sb, flags, 0, 1);

BITMASK(BDEV_WRITEBACK,	struct cache_sb, flags, 0, 1);
BITMASK(BDEV_STATE,	struct cache_sb, flags, 61, 2);
#define BDEV_STATE_NONE		0U
#define BDEV_STATE_CLEAN	1U
#define BDEV_STATE_DIRTY	2U
#define BDEV_STATE_STALE	3U

struct gc_stat {
	unsigned		count;
	unsigned		ms_max;
	time_t			last;

	size_t			nodes;
	size_t			key_bytes;

	size_t			nkeys;
	uint64_t		data;
	uint64_t		dirty;
};

struct cache_set {
	struct list_head	list;
	struct cache_sb		sb;

	struct cache		*cache[MAX_CACHES_PER_SET];
	struct cache		*cache_by_alloc[MAX_CACHES_PER_SET];
	int			caches_loaded;

	atomic_t		closing;
	struct kobject		kobj;
	struct work_struct	unregister;
	struct list_head	devices;

	struct mutex		sb_write;
	struct closure		*sb_writer;

	struct bio_set		*bio_split;
	struct shrinker		shrink;

	/*
	 * Buckets used for cached data go on the heap. The heap is ordered by
	 * bucket->priority; a priority of ~0 indicates a btree bucket. Priority
	 * is increased on cache hit, and periodically all the buckets on the
	 * heap have their priority scaled down by a linear function.
	 */
	spinlock_t		bucket_lock;
	int			bucket_bits;
	int			btree_pages;

	/* Refcount for when we can't write the priorities to disk until a
	 * btree write finishes.
	 */
	atomic_t		prio_blocked;
	closure_list_t		bucket_wait;

	unsigned		rescale_value;
	atomic_t		rescale;
	uint16_t		min_prio;
	uint8_t			need_gc;
	struct gc_stat		gc_stats;

	struct list_head	lru;
	struct list_head	freed;
	struct closure		*try_harder;
	closure_list_t		try_wait;

	struct work_struct	gc_work;
	struct mutex		gc_lock;
	struct bkey		gc_done;
	atomic_t		sectors_to_gc;

	struct btree		*root;

	int			nr_uuids;
	struct uuid_entry	*uuids;
	BKEY_PADDED(uuid_bucket);
	struct closure		*uuid_writer;

	struct mutex		fill_lock;
	struct btree_iter	*fill_iter;

	struct mutex		sort_lock;
	struct bset		*sort;

	struct list_head	open_buckets;
	struct list_head	dirty_buckets;
	spinlock_t		open_bucket_lock;

	struct journal		journal;

	atomic_long_t		writeback_keys_done;
	atomic_long_t		writeback_keys_failed;
	atomic_long_t		btree_write_count;
	atomic_long_t		keys_write_count;

#define BUCKET_HASH_BITS	7
	struct hlist_head	bucket_hash[];
};

struct cache {
	struct cache_set	*set;
	struct cache_sb		sb;
	struct bio		sb_bio;
	struct bio_vec		sb_bv[1];

	struct kobject		kobj;
	struct block_device	*bdev;
	struct dentry		*debug;

	struct bucket		*buckets;

	struct bio		*uuid_bio;

	DECLARE_HEAP(struct bucket *, heap);

	struct closure		prio;
	struct bio		*prio_bio;
	struct prio_set		*disk_buckets;

	/* Bucket that journal uses */
	uint64_t		prio_start;

	uint64_t		*prio_buckets;
	uint64_t		*prio_next;
	int			prio_write;
	int			prio_alloc;

	/* > 0: buckets in free_inc have been marked as free
	 * = 0: buckets in free_inc can't be used until priorities are written
	 * < 0: priority write in progress
	 */
	atomic_t		prio_written;
	uint8_t			need_save_prio;

	DECLARE_FIFO(long, free);
	DECLARE_FIFO(long, free_inc);
	DECLARE_FIFO(long, btree_freed);

	atomic_long_t		meta_sectors_written;
	atomic_long_t		btree_sectors_written;
	atomic_long_t		sectors_written;

	bool			discard;
	struct list_head	discards;
	struct page		*discard_page;

	sector_t		journal_area_start;
	sector_t		journal_area_end;

	/* Free journal sectors */
	sector_t		journal_start;
	sector_t		journal_end;
	DECLARE_FIFO(struct journal_seq, journal);
	struct bio		journal_bio;
	struct bio_vec		journal_bv[8];
};

struct open_bucket {
	struct list_head	list;
	struct task_struct	*last;
	unsigned		sectors_free;
	BKEY_PADDED(key);
};

struct dirty_io {
	struct closure		cl;
	struct cached_dev	*d;
	struct bio		bio;
};

struct dirty {
	struct rb_node		node;
	BKEY_PADDED(key);
	struct dirty_io		*io;
};

#define RECENT_IO_BITS	7
#define RECENT_IO	(1 << RECENT_IO_BITS)

struct io {
	/* Used to track sequential IO so it can be skipped */
	struct hlist_node	hash;
	struct list_head	lru;

	unsigned long		jiffies;
	unsigned		sequential;
	sector_t		last;
};

struct cached_dev {
	struct list_head	list;
	struct cache_sb		sb;
	struct bio		sb_bio;
	struct bio_vec		sb_bv[1];

	struct closure		*sb_writer;
	struct semaphore	sb_write;

	struct kobject		kobj;
	struct block_device	*bdev;
	struct gendisk		*disk;

	struct cache_set	*c;
	unsigned		id;

	spinlock_t		lock;
	/* Refcount on the cache set. Always nonzero when we're caching. */
	atomic_t		count;
	atomic_t		unregister;
	/* If nonzero, we're disabling caching */
	atomic_t		closing;
	atomic_t		running;

	struct rw_semaphore	writeback_lock;
	struct work_struct	refill;
	mempool_t		*search;

	atomic_long_t		sectors_bypassed;
	atomic_long_t		cache_hits;
	atomic_long_t		cache_misses;
	atomic_long_t		cache_readaheads;
	atomic_long_t		cache_miss_collisions;

	unsigned long		sequential_cutoff_average;
	unsigned long		sequential_cutoff;
	unsigned		sequential_merge:1;

	unsigned		writeback:1;
	unsigned		writeback_running:1;
	unsigned short		writeback_percent;
	unsigned		writeback_delay;

	unsigned long		readahead;

	atomic_t		in_flight;
	atomic_long_t		last_refilled;
	uint64_t		last_found;
	uint64_t		last_read;
	struct rb_root		dirty;

	struct io		io[RECENT_IO];
	struct hlist_head	io_hash[RECENT_IO + 1];
	struct list_head	io_lru;
};

struct btree_write {
#ifdef DEBUG_LATENCY
	unsigned long		wait_time;
#endif
	struct btree		*b;
	closure_list_t		wait;
	struct closure		*owner;
	atomic_t		*journal;

	int			prio_blocked;
	bool			nofree;
};

struct btree {
	struct list_head	lru;
	struct hlist_node	hash;
	struct rw_semaphore	lock;
	struct delayed_work	work;

	unsigned long		jiffies;

	struct cache_set	*c;
	closure_list_t		wait;

	unsigned long		expires;
	struct btree_write	*write;
	atomic_t		io;
	int			prio_blocked;

	struct btree_write	writes[2];

	atomic_t		nread;
	short			level;
	uint16_t		written;
	uint16_t		nsets;
	unsigned		next:1;
	unsigned		page_order:7;

	BKEY_PADDED(key);

	union {
		struct bset		*data;
		struct bset		*sets[8];
	};
	struct bio		bio;
};

struct keylist {
	struct bkey		*top;
	union {
		uint64_t		*list;
		struct bkey		*bottom;
	};

	/* Enough room for btree_split's keys without realloc */
#define KEYLIST_INLINE		16
	uint64_t		d[KEYLIST_INLINE];
};

struct search {
	/* Stack frame for bio_complete */
	struct closure		cl;

	struct cached_dev	*d;
	struct task_struct	*task;

	struct bio		*bio;
	struct bio		*cache_bio;

	/* Original bi_end_io and bi_private from s->bio */
	bio_end_io_t		*bi_end_io;
	void			*bi_private;

	/* Stack frame for btree code */
	struct closure		insert;

	/* For cache lookups, keys we took refcounts on.
	 * Everywhere else, keys to be inserted.
	 */
	struct keylist		keys;

	/* Journal entry we have a refcount on */
	atomic_t		*journal;

	/* IO error returned to s->bio */
	short			error;

	/* Starting vec in cache_bio after which we must free pages */
	unsigned short		pages_from;

	/* Btree level at which we start taking write locks */
	short			lock;

	/* Btree insertion type */
	enum {
		INSERT_READ		= 0,
		INSERT_WRITE		= 1,
		INSERT_WRITEBACK	= 3,
		INSERT_UNDIRTY		= 4,
		INSERT_REPLAY		= 6
	} insert_type:8;

	unsigned		skip:1;
	unsigned		bio_done:1;
	unsigned		lookup_done:1;
	unsigned		cache_hit:1;
};

const char *insert_types[] = {
	"read", "write", NULL, "writeback",
	"undirty", NULL, "replay"
}

#define insert_type(s)	insert_types[s->insert_type]

#define CREATE_TRACE_POINTS
#include <trace/events/bcache.h>

EXPORT_TRACEPOINT_SYMBOL_GPL(bcache_start_closure_wait);
EXPORT_TRACEPOINT_SYMBOL_GPL(bcache_end_closure_wait);

static struct kobject *bcache_kobj;
static struct mutex register_lock;
static LIST_HEAD(uncached_devices);
static LIST_HEAD(cache_sets);

static struct kmem_cache *search_cache, *dirty_cache;
static struct workqueue_struct *delayed, *writeback;
static int bcache_major, bcache_minor;

/*
 * Sysfs vars / tunables
 */
static uint16_t	initial_prio = 32768;
static unsigned latency_warn_ms;

static void dump_bucket_and_panic(struct btree *, const char *, ...);
static int __btree_write(struct btree *);
static void __btree_sort(struct btree *, int, struct bset *,
			 struct btree_iter *, bool);
static void read_dirty(struct cached_dev *);
static void queue_writeback(struct cached_dev *);
static void prio_write(struct cache *, struct closure *);
static void write_bdev_super(struct cached_dev *, struct closure *);
static bool cache_set_error(struct cache_set *, const char *, ...);
static void do_discard(struct cache *);
static void cache_request_endio(struct bio *, int);
static void __request_read(struct closure *);
static void __bio_complete(struct search *);
static void btree_journal(struct closure *);
static bool journal_full(struct cache_set *c);
static void bio_insert(struct closure *);
static void btree_invalidate(struct search *);
static void set_new_root(struct btree *);
static void btree_journal_wait(struct cache_set *, struct closure *);
static inline void cached_dev_put(struct cached_dev *);
static void cache_init_journal(struct cache *);

#define btree_prio		((uint16_t) ~0)
#define MAX_NEED_GC		64
#define MAX_SAVE_PRIO		72

#define CUTOFF_CACHE_ADD	95
#define CUTOFF_CACHE_READA	90
#define CUTOFF_WRITEBACK	50
#define CUTOFF_WRITEBACK_SYNC	75

#define BTREE_MAX_PAGES		(256 * 1024 / PAGE_SIZE)

#define btree_reserve(c)	((c->root ? c->root->level : 1) * 4 + 4)

#define btree_bytes(c)		((c)->btree_pages * PAGE_SIZE)
#define btree_blocks(b)		(KEY_SIZE(&b->key) / (b)->c->sb.block_size)

#define bucket_pages(c)		((c)->sb.bucket_size / PAGE_SECTORS)
#define bucket_bytes(c)		((c)->sb.bucket_size << 9)
#define block_bytes(c)		((c)->sb.block_size << 9)

#define prios_per_bucket(c)				\
	((bucket_bytes(c) - sizeof(struct prio_set)) /	\
	 sizeof(struct bucket_disk))
#define prio_buckets(c)					\
	DIV_ROUND_UP((c)->sb.nbuckets, prios_per_bucket(c))

#define JSET_MAGIC	0x245235c1a3625032
#define PSET_MAGIC	0x6750e15f87337f91
#define BSET_MAGIC	0x90135c78b99e07f5

#define jset_magic(c)		((c)->sb.set_magic ^ JSET_MAGIC)
#define pset_magic(c)		((c)->sb.set_magic ^ PSET_MAGIC)
#define bset_magic(c)		((c)->sb.set_magic ^ BSET_MAGIC)

#define bucket_to_sector(c, b)	(((sector_t) (b)) << c->bucket_bits)
#define sector_to_bucket(c, s)	((long) (s >> c->bucket_bits))

#define __set_bytes(i, k)	(sizeof(*(i)) + (k) * sizeof(uint64_t))
#define set_bytes(i)		__set_bytes(i, i->keys)

#define __set_blocks(i, k, c)	DIV_ROUND_UP(__set_bytes(i, k), block_bytes(c))
#define set_blocks(i, c)	__set_blocks(i, i->keys, c)

#define node(i, j)		((struct bkey *) ((i)->d + (j)))
#define end(i)			node(i, (i)->keys)
#define last_key(i)		(i->keys ? prev(node(i, (i)->keys)) : NULL)

#define csum_set(i)							\
	crc64(((void *) (i)) + 8, ((void *) end(i)) - (((void *) (i)) + 8))

#define index(i, b)							\
	((int) (((void *) i - (void *) (b)->data) / block_bytes(b->c)))

/* Btree key macros */

#define KEY_FIELD(name, field, offset, size)				\
	static inline uint64_t name(const struct bkey *k)		\
	{ return (k->field >> offset) & ~(((uint64_t) ~0) << size); }	\
									\
	static inline void SET_##name(struct bkey *k, uint64_t v)	\
	{								\
		k->field &= ~(~((uint64_t) ~0 << size) << offset);	\
		k->field |= v << offset;				\
	}

/* All units are in sectors */
KEY_FIELD(KEY_PTRS,	header, 60, 3)
KEY_FIELD(KEY_DIRTY,	header, 36, 1)
KEY_FIELD(KEY_SIZE,	header, 20, 16)
KEY_FIELD(KEY_DEV,	header, 0,  20)
KEY_FIELD(KEY_SECTOR,	key,	16, 47)
KEY_FIELD(KEY_SNAPSHOT,	key,	0,  16)

#define KEY_HEADER(len, dev)						\
	(((uint64_t) 1 << 63) | ((uint64_t) (len) << 20) | dev)

#define KEY(dev, sector, len)	(struct bkey)				\
	{ .header = KEY_HEADER(len, dev), .key = sector}

#define KEY_START(k)		((k)->key - KEY_SIZE(k))
#define START_KEY(k)		KEY(KEY_DEV(k), KEY_START(k), 0)
#define MAX_KEY			KEY(~(~0 << 20), ((uint64_t) ~0) >> 1, 0)
#define ZERO_KEY		KEY(0, 0, 0)

#define KEY_IS_HEADER(k)	((k)->header >> 63)
#define PTR_DIRTY_BIT		(((uint64_t) 1 << 36))

#define PTR(gen, offset, dev)						\
	((((uint64_t) dev) << 51) | ((uint64_t) offset) << 8 | gen)

#define PTR_DEV(k, n)		((k)->ptr[n] >> 51)
#define PTR_OFFSET(k, n)	(((k)->ptr[n] >> 8) & ~((int64_t) ~0 << 51))
#define PTR_GEN(k, n)		((uint8_t) ((k)->ptr[n]) & 255)
#define PTR_HASH(k)		((k)->ptr[0] >> 8)

#define PTR_CACHE(c, k, n)	(c->cache[PTR_DEV(k, n)])
#define PTR_BUCKET_NR(c, k, n)	sector_to_bucket(c, PTR_OFFSET(k, n))

#define PTR_BUCKET(c, k, n)						\
	(PTR_CACHE(c, k, n)->buckets + PTR_BUCKET_NR(c, k, n))

#define bio_split_c(bio, len, c)					\
	bio_split_front(bio, len, GFP_NOIO, (c)->bio_split)

/* Error handling macros */

#define cache_bug(b, ...)						\
do {									\
	if (__builtin_types_compatible_p(typeof(b), struct cache *)	\
		? cache_error(((struct cache *) b), __VA_ARGS__)	\
		: cache_set_error(((struct btree *) b)->c, __VA_ARGS__))\
		dump_stack();						\
} while (0)

#define cache_set_err_on(cond, c, ...)					\
	({ if (cond) cache_set_error(c, __VA_ARGS__); })

#define cache_err_on(cond, c, ...)					\
	({ if (cond) cache_error(c, __VA_ARGS__); })

#define cache_bug_on(cond, c, ...)					\
	({ if (cond) cache_bug(c, __VA_ARGS__); })

#define cache_error(c, ...)	cache_set_error(c->set, __VA_ARGS__)

#define err_printk(...)	printk(KERN_ERR "bcache: " __VA_ARGS__)

/* Looping macros */

#define for_each_cache(c, s)						\
	for (int _i = 0; c = s->cache[_i], _i < s->sb.nr_in_set; _i++)

#define for_each_bucket(b, c)						\
	for (b = (c)->buckets + (c)->sb.first_bucket;			\
	     b < (c)->buckets + (c)->sb.nbuckets; b++)

#define for_each_sorted_set_start(b, i, start)				\
	for (int _i = start; i = b->sets[_i], _i <= b->nsets; _i++)

#define for_each_sorted_set(b, i)	for_each_sorted_set_start(b, i, 0)

#define bkey_filter(b, i, k, filter)					\
({									\
	while (k < end(i) && filter(b, k))				\
		k = next(k);						\
	k;								\
})

#define all_keys(b, k)		0

#define for_each_key_after_filter(b, k, search, filter)			\
	for (struct bset **_i = b->sets; _i <= b->sets + b->nsets; _i++)\
		for (k = btree_bsearch(*_i, search);			\
		     (k = bkey_filter(b, *_i, k, filter)) < end(*_i);	\
		     k = next(k))

#define for_each_key_filter(b, k, filter)				\
	for_each_key_after_filter(b, k, NULL, filter)

#define for_each_key(b, k)	for_each_key_filter(b, k, all_keys)

static struct bset *write_block(struct btree *b)
{
	 return ((void *) b->data) + b->written * block_bytes(b->c);
}

static inline void rw_lock(bool w, struct btree *b, int level)
{
	w ? down_write_nested(&b->lock, level + 1)
	  : down_read_nested(&b->lock, level + 1);
}

static inline void __rw_unlock(bool w, struct btree *b, bool nowrite)
{
	bool queue;
	long delay = max_t(long, 0, b->expires - jiffies);
	BUG_ON(!b->written && atomic_read(&b->nread) == 1 && b->data->keys);

	if (!delay && !nowrite)
		__btree_write(b);

	queue = b->write;

	(w ? up_write : up_read)(&b->lock);

	if (queue) {
		smp_rmb();
		if (atomic_read(&b->io) == -1)
			schedule_delayed_work(&b->work, delay);
	}
}

#define rw_unlock_nowrite(w, b)	__rw_unlock(w, b, true)
#define rw_unlock(w, b)		__rw_unlock(w, b, false)

/* Btree key comparison/iteration */

static inline size_t key_bytes(const struct bkey *k)
{
	return (2 + KEY_PTRS(k)) * sizeof(uint64_t);
}

static int64_t bkey_cmp(const struct bkey *l, const struct bkey *r)
{
	return (int64_t) KEY_DEV(l) - (int64_t) KEY_DEV(r)
		?: (int64_t) l->key - (int64_t) r->key;
}

__pure
static struct bkey *next(const struct bkey *k)
{
	uint64_t *d = (void *) k;
	return (struct bkey *) (d + 2 + KEY_PTRS(k));
}

__pure
static struct bkey *prev(const struct bkey *k)
{
	uint64_t *d = (void *) k;
	do {
		--d;
	} while (!KEY_IS_HEADER((struct bkey *) d));

	return (struct bkey *) d;
}

__pure
static struct bkey *btree_bsearch(struct bset *i, const struct bkey *search)
{
	/* Returns the smallest key greater than the search key.
	 * This is because we index by the end, not the beginning
	 */
	int l = 0, r = i->keys;

	if (search)
		while (l < r) {
			int m = (l + r) >> 1;
			while (!KEY_IS_HEADER(node(i, m)))
				m--;

			if (m == l && next(node(i, m)) != node(i, r))
				m += 2 + KEY_PTRS(node(i, m));

			if (bkey_cmp(node(i, m), search) > 0)
				r = m;
			else
				l = m + 2 + KEY_PTRS(node(i, m));
		}

	return node(i, l);
}

/* Btree iterator */

struct btree_iter {
	struct btree_iter_set {
		struct bkey *k, *end;
	} *top, sets[8];
};

static bool btree_iter_end(struct btree_iter *iter)
{
	return iter->top < iter->sets;
}

static void __btree_iter_bubble(struct btree_iter *iter,
				struct btree_iter_set *start)
{
	int64_t cmp(struct bkey *l, struct bkey *r)
	{
		return bkey_cmp(&START_KEY(l), &START_KEY(r));
	}

	for (struct btree_iter_set *i = start - 1;
	     i >= iter->sets && cmp(i[1].k, i[0].k) > 0;
	     --i)
		swap(i[1], i[0]);
}

static void btree_iter_bubble(struct btree_iter *iter)
{
	__btree_iter_bubble(iter, iter->top);
}

static struct bkey *btree_iter_init(struct btree *b, struct btree_iter *iter,
				    struct bkey *search, int start)
{
	struct bkey *ret = NULL;
	struct bset *i;
	iter->top = iter->sets;

	for_each_sorted_set_start(b, i, start) {
		iter->top->k	= btree_bsearch(i, search);
		iter->top->end	= end(i);
		ret = iter->top->k;

		if (iter->top->k != iter->top->end) {
			btree_iter_bubble(iter);
			iter->top++;
		}
	}

	iter->top--;
	return ret;
}

static struct bkey *btree_iter_next(struct btree_iter *iter)
{
	struct bkey *ret = NULL;

	if (!btree_iter_end(iter)) {
		ret = iter->top->k;
		iter->top->k = next(iter->top->k);

		if (iter->top->k > iter->top->end) {
			__WARN();
			iter->top->k = iter->top->end;
		}

		if (iter->top->k == iter->top->end)
			iter->top--;

		btree_iter_bubble(iter);
	}

	return ret;
}

/* Btree key manipulation */

static void bkey_copy_key(struct bkey *dest, const struct bkey *src)
{
	if (!src)
		src = &KEY(0, 0, 0);

	SET_KEY_DEV(dest, KEY_DEV(src));
	dest->key = src->key;
}

static void bkey_copy(struct bkey *dest, const struct bkey *src)
{
	memcpy(dest, src, key_bytes(src));
}

static bool __cut_front(const struct bkey *where, struct bkey *k)
{
	int len = 0;

	if (bkey_cmp(where, &START_KEY(k)) <= 0)
		return false;

	if (bkey_cmp(where, k) < 0)
		len = k->key - where->key;
	else
		bkey_copy_key(k, where);

	for (unsigned i = 0; i < KEY_PTRS(k); i++)
		k->ptr[i] += PTR(0, KEY_SIZE(k) - len, 0);

	BUG_ON(len > KEY_SIZE(k));
	SET_KEY_SIZE(k, len);
	return true;
}

static bool cut_front(const struct bkey *where, struct bkey *k)
{
	BUG_ON(bkey_cmp(where, k) > 0);
	return __cut_front(where, k);
}

static bool __cut_back(const struct bkey *where, struct bkey *k)
{
	int len = 0;

	if (bkey_cmp(where, k) >= 0)
		return false;

	BUG_ON(KEY_DEV(where) != KEY_DEV(k));

	if (bkey_cmp(where, &START_KEY(k)) > 0)
		len = where->key - KEY_START(k);

	bkey_copy_key(k, where);

	BUG_ON(len > KEY_SIZE(k));
	SET_KEY_SIZE(k, len);
	return true;
}

static bool cut_back(const struct bkey *where, struct bkey *k)
{
	BUG_ON(bkey_cmp(where, &START_KEY(k)) < 0);
	return __cut_back(where, k);
}

static void __bkey_put(struct cache_set *c, struct bkey *k)
{
	for (unsigned i = 0; i < KEY_PTRS(k); i++)
		atomic_dec_bug(&PTR_BUCKET(c, k, i)->pin);
}

static void bkey_put(struct cache_set *c, struct bkey *k, int write, int level)
{
	if ((level && k->key) ||
	    (!level && write != INSERT_UNDIRTY))
		__bkey_put(c, k);
}

static void bset_init(struct btree *b, struct bset *i)
{
	if (i == b->data)
		get_random_bytes(&b->data->seq, sizeof(uint64_t));

	i->magic	= bset_magic(b->c);
	i->seq		= b->data->seq;
	i->version	= 0;
	i->keys		= 0;
}

/* Btree/bkey debug printing */

struct keyprint_hack {
	char s[40];
};

static struct keyprint_hack _pkey(const struct bkey *k)
{
	struct keyprint_hack r;
	int i = scnprintf(r.s, 40, "%llu:%llu len %llu -> ",
			 KEY_DEV(k), k->key, KEY_SIZE(k));

	if (KEY_PTRS(k))
		i += scnprintf(r.s + i, 40 - i, "%llu gen %i",
			      PTR_OFFSET(k, 0), PTR_GEN(k, 0));
	else
		i += scnprintf(r.s + i, 40 - i, "[]");

	if (KEY_DIRTY(k))
		scnprintf(r.s + i, 40 - i, " dirty");
	return r;
}

#define pkey(k)		(_pkey(k).s)

static struct keyprint_hack _pbtree(const struct btree *b)
{
	struct keyprint_hack r;

	snprintf(r.s, 40, "%li level %i/%i", PTR_BUCKET_NR(b->c, &b->key, 0),
		 b->level, b->c->root ? b->c->root->level : -1);
	return r;
}

#define pbtree(b)	(_pbtree(b).s)

/* Keylists */

static void keylist_init(struct keylist *l)
{
	l->top = (void *) (l->list = l->d);
}

static void keylist_free(struct keylist *l)
{
	if (l->list != l->d)
		kfree(l->list);
}

static void keylist_copy(struct keylist *dest, struct keylist *src)
{
	*dest = *src;

	if (src->list == src->d) {
		size_t n = (uint64_t *) src->top - src->d;
		dest->top = (struct bkey *) &dest->d[n];
		dest->list = dest->d;
	}
}

static int keylist_realloc(struct keylist *l, int nptrs)
{
	unsigned n = (uint64_t *) l->top - l->list;
	unsigned size = roundup_pow_of_two(n + 2 + nptrs);
	uint64_t *new;

	if (size <= KEYLIST_INLINE ||
	    roundup_pow_of_two(n) == size)
		return 0;

	new = krealloc(l->list == l->d ? NULL : l->list,
		       sizeof(uint64_t) * size, GFP_NOIO);

	if (!new)
		return -ENOMEM;

	if (l->list == l->d)
		memcpy(new, l->list, sizeof(uint64_t) * KEYLIST_INLINE);

	l->list = new;
	l->top = (struct bkey *) (&l->list[n]);

	return 0;
}

static struct bkey *keylist_pop(struct keylist *l)
{
	if (l->top == (struct bkey *) l->list)
		return NULL;

	l->top = prev(l->top);
	BUG_ON((uint64_t *) l->top < l->list);

	return l->top;
}

static void keylist_push(struct keylist *l)
{
	BUG_ON(!KEY_IS_HEADER(l->top));
	l->top = next(l->top);
}

static void keylist_add(struct keylist *l, struct bkey *k)
{
	bkey_copy(l->top, k);
	keylist_push(l);
}

static bool keylist_empty(struct keylist *l)
{
	return l->top == (void *) l->list;
}

static void search_init(struct search *s)
{
	memset(s, 0, sizeof(struct search));
	keylist_init(&s->keys);

	s->lock		= -1;
}

static void search_init_stack(struct search *s)
{
	search_init(s);
	closure_init_stack(&s->insert);
}

/* Bucket heap / gen */

__pure
static inline unsigned in_use(const struct cache_set *s)
{
	struct cache *c;
	uint64_t nbuckets = 0, heap = 0;

	for_each_cache(c, s)
		nbuckets += c->sb.nbuckets, heap += c->heap.size;

	return div64_u64((nbuckets - heap) * 100, nbuckets);
}

#define bucket_gc_gen(b)	((uint8_t) ((b)->gen - (b)->last_gc))
#define bucket_disk_gen(b)	((uint8_t) ((b)->gen - (b)->disk_gen))

static uint8_t inc_gen(struct cache *c, struct bucket *b)
{
	uint8_t ret = ++b->gen;

	c->set->need_gc = max(c->set->need_gc, bucket_gc_gen(b));
	BUG_ON(c->set->need_gc > 97);

	if (CACHE_SYNC(&c->set->sb)) {
		c->need_save_prio = max(c->need_save_prio, bucket_disk_gen(b));
		BUG_ON(c->need_save_prio > 96);
	}

	return ret;
}

static void rescale_heap(struct cache_set *s, int sectors)
{
	struct cache *c;
	struct bucket *b;
	int r;

	atomic_sub(sectors, &s->rescale);

	do {
		r = atomic_read(&s->rescale);

		if (r >= 0)
			return;
	} while (atomic_cmpxchg(&s->rescale, r, r + s->rescale_value) != r);

	spin_lock(&s->bucket_lock);

	for_each_cache(c, s)
		for_each_bucket(b, c)
			if (b->prio &&
			    b->prio != btree_prio &&
			    !atomic_read(&b->pin)) {
				b->prio--;
				s->min_prio = min(s->min_prio, b->prio);
			}

	spin_unlock(&s->bucket_lock);
}

static inline void bucket_add_heap(struct cache *c, struct bucket *b)
{
	if (!b->mark)
		b->prio = 0;

	if (b->mark >= 0 &&
	    bucket_gc_gen(b) < 96U) {
		if (!b->mark &&
		    bucket_disk_gen(b) < 64U &&
		    fifo_push(&c->btree_freed, b - c->buckets))
			atomic_inc(&b->pin);
		else if (b->prio != btree_prio)
			heap_add(&c->heap, b, heap, bucket_cmp);
	}
}

static long pop_heap(struct cache *c)
{
	/* On cache hit, priority is increased but we don't readjust
	 * the heap so as not to take the lock there - hence the heap
	 * isn't necessarily a heap. This mostly works provided priority
	 * only goes up - later we won't keep the full heap around
	 * which will be better.
	 */

	while (1) {
		struct bucket *b = heap_peek(&c->heap);

		if (!b) {
			queue_work(delayed, &c->set->gc_work);
			printk_ratelimited(KERN_WARNING
					   "bcache: heap empty!\n");
			return -1;
		}

		heap_sift(&c->heap, 0, heap, bucket_cmp);
		b = heap_pop(&c->heap, heap, bucket_cmp);

		if (bucket_disk_gen(b) >= 96U)
			continue;

		inc_gen(c, b);

		smp_mb();
		if (atomic_read(&b->pin))
			continue;

		b->prio = initial_prio;
		atomic_inc(&b->pin);

		return b - c->buckets;
	}
}

static long pop_freed(struct cache *c)
{
	long r;

	if ((!CACHE_SYNC(&c->set->sb) ||
	     !atomic_read(&c->set->prio_blocked)) &&
	    fifo_pop(&c->btree_freed, r))
		return r;

	if ((!CACHE_SYNC(&c->set->sb) ||
	     atomic_read(&c->prio_written) > 0) &&
	    fifo_pop(&c->free_inc, r))
		return r;

	return  !CACHE_SYNC(&c->set->sb)
		? pop_heap(c) : -1;
}

/* Discard/TRIM */

static void discard_finish(struct work_struct *w)
{
	struct discard *d = container_of(w, struct discard, work);
	struct cache *c = d->c;
	bool run = false;

	spin_lock(&c->set->bucket_lock);
	if (fifo_empty(&c->free) ||
	    fifo_used(&c->free) == 8)
		run = true;

	fifo_push(&c->free, d->bucket);

	list_add(&d->list, &c->discards);

	do_discard(c);
	spin_unlock(&c->set->bucket_lock);

	if (run)
		closure_run_wait(&c->set->bucket_wait, NULL);
}

static void discard_endio(struct bio *bio, int error)
{
	struct discard *d = container_of(bio, struct discard, bio);

	if (error) {
		printk(KERN_NOTICE "bcache: discard error, disabling\n");
		d->c->discard = 0;
	}

	PREPARE_WORK(&d->work, discard_finish);
	schedule_work(&d->work);
}

static void discard_work(struct work_struct *w)
{
	struct discard *d = container_of(w, struct discard, work);
	submit_bio(0, &d->bio);
}

static void do_discard(struct cache *c)
{
	struct request_queue *q = bdev_get_queue(c->bdev);
	int s = q->limits.logical_block_size;

	while (c->discard &&
	       !list_empty(&c->discards) &&
	       fifo_free(&c->free) >= 8) {
		struct discard *d = list_first_entry(&c->discards,
						     struct discard, list);

		d->bucket = pop_freed(c);
		if (d->bucket == -1)
			break;

		list_del(&d->list);

		bio_init(&d->bio);
		memset(&d->bv, 0, sizeof(struct bio_vec));
		bio_set_prio(&d->bio, IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0));

		d->bio.bi_sector	= bucket_to_sector(c->set, d->bucket);
		d->bio.bi_bdev		= c->bdev;
		d->bio.bi_rw		= DISCARD_NOBARRIER;
		d->bio.bi_max_vecs	= 1;
		d->bio.bi_io_vec	= d->bio.bi_inline_vecs;
		d->bio.bi_end_io	= discard_endio;

		if (bio_add_pc_page(q, &d->bio, c->discard_page, s, 0) < s) {
			printk(KERN_DEBUG "bcache: bio_add_pc_page failed\n");
			c->discard = 0;
			fifo_push(&c->free, d->bucket);
			list_add(&d->list, &c->discards);
			break;
		}

		d->bio.bi_size = bucket_bytes(c);

		PREPARE_WORK(&d->work, discard_work);
		schedule_work(&d->work);
	}
}

/* Allocation */

static void free_some_buckets(struct cache *c)
{
	long r;

	do_discard(c);

	while (!fifo_full(&c->free) &&
	       (fifo_used(&c->free) <= 8 ||
		!c->discard) &&
	       (r = pop_freed(c)) != -1)
		fifo_push(&c->free, r);

	while (c->prio_alloc != prio_buckets(c) &&
	       fifo_pop(&c->free, r)) {
		struct bucket *b = c->buckets + r;
		c->prio_next[c->prio_alloc++] = r;

		b->mark = GC_MARK_BTREE;
		atomic_dec_bug(&b->pin);
	}

	if (!CACHE_SYNC(&c->set->sb))
		return;

	/* XXX: tracepoint for when c->need_save_prio > 64 */

	if (atomic_read(&c->prio_written) > 0 &&
	    (fifo_empty(&c->free_inc) ||
	     c->need_save_prio > 64))
		atomic_set(&c->prio_written, 0);
	else if (atomic_read(&c->prio_written))
		return;

	while (!fifo_full(&c->free_inc) &&
	       ((r = pop_heap(c)) != -1))
		fifo_push(&c->free_inc, r);

	if (c->heap.size * 8 < c->sb.nbuckets)
		queue_work(delayed, &c->set->gc_work);

	if (atomic_read(&c->set->prio_blocked))
		return;

	if (fifo_full(&c->free_inc) ||
	    c->need_save_prio > 64 ||
	    (!c->heap.size && !fifo_empty(&c->free_inc)))
		prio_write(c, NULL);
}

static long pop_bucket(struct cache *c, uint16_t priority, struct closure *cl)
{
	long r = -1;
again:
	free_some_buckets(c);

	if ((priority == btree_prio ||
	     fifo_used(&c->free) > 8) &&
	    fifo_pop(&c->free, r)) {
		struct bucket *b = c->buckets + r;
#ifdef CONFIG_BCACHE_EDEBUG
		long i;
		for (i = 0; i < prio_buckets(c); i++)
			BUG_ON(c->prio_buckets[i] == r);
		for (i = 0; i < c->prio_alloc; i++)
			BUG_ON(c->prio_next[i] == r);

		fifo_for_each(i, &c->free)
			BUG_ON(i == r);
		fifo_for_each(i, &c->free_inc)
			BUG_ON(i == r);
		fifo_for_each(i, &c->btree_freed)
			BUG_ON(i == r);
#endif
		BUG_ON(atomic_read(&b->pin) != 1);
		BUG_ON(b->heap != -1);

		b->prio = priority;
		b->mark = priority == btree_prio
			? GC_MARK_BTREE
			: c->sb.bucket_size;
		return r;
	}

	pr_debug("no free buckets, prio_written %i, blocked %i, "
		 "free %zu, free_inc %zu, btree_freed %zu",
		 atomic_read(&c->prio_written),
		 atomic_read(&c->set->prio_blocked), fifo_used(&c->free),
		 fifo_used(&c->free_inc), fifo_used(&c->btree_freed));

	if (cl) {
		if (test_bit(CLOSURE_BLOCK, &cl->flags))
			spin_unlock(&c->set->bucket_lock);

		closure_wait_on(&c->set->bucket_wait, delayed, cl,
				atomic_read(&c->prio_written) > 0 ||
				(!atomic_read(&c->set->prio_blocked) &&
				 !atomic_read(&c->prio_written)));

		if (test_bit(CLOSURE_BLOCK, &cl->flags)) {
			spin_lock(&c->set->bucket_lock);
			goto again;
		}
	}

	return -1;
}

static void unpop_bucket(struct cache_set *c, struct bkey *k)
{
	for (unsigned i = 0; i < KEY_PTRS(k); i++) {
		struct bucket *b = PTR_BUCKET(c, k, i);

		b->mark = 0;
		bucket_add_heap(PTR_CACHE(c, k, i), b);
	}
}

static int __pop_bucket_set(struct cache_set *c, uint64_t prio,
			    struct bkey *k, int n, struct closure *cl)
{
	lockdep_assert_held(&c->bucket_lock);
	BUG_ON(!n || n > c->caches_loaded || n > 8);

	k->header = KEY_HEADER(0, 0);

	/* sort by free space/prio of oldest data in caches */

	for (int i = 0; i < n; i++) {
		struct cache *ca = c->cache_by_alloc[i];
		long b = pop_bucket(ca, prio, cl);

		if (b == -1)
			goto err;

		k->ptr[i] = PTR(ca->buckets[b].gen,
				bucket_to_sector(c, b),
				ca->sb.nr_this_dev);

		SET_KEY_PTRS(k, i + 1);
	}

	return 0;
err:
	unpop_bucket(c, k);
	__bkey_put(c, k);
	return -1;
}

static int pop_bucket_set(struct cache_set *c, uint64_t prio,
			  struct bkey *k, int n, struct closure *cl)
{
	int ret;
	spin_lock(&c->bucket_lock);
	ret = __pop_bucket_set(c, prio, k, n, cl);
	spin_unlock(&c->bucket_lock);
	return ret;
}

static inline uint8_t gen_after(uint8_t a, uint8_t b)
{
	uint8_t r = a - b;
	return r > 128U ? 0 : r;
}

#define ptr_stale(c, k, n)					\
	gen_after(PTR_BUCKET(c, k, n)->gen, PTR_GEN(k, n))

static const char *ptr_status(struct cache_set *c, const struct bkey *k)
{
	for (unsigned i = 0; i < KEY_PTRS(k); i++) {
		struct cache *ca = PTR_CACHE(c, k, i);
		size_t bucket = PTR_BUCKET_NR(c, k, i);
		size_t r = PTR_OFFSET(k, i) & ~(~0 << c->bucket_bits);

		if (PTR_DEV(k, i) > MAX_CACHES_PER_SET)
			return "bad cache device";
		if (KEY_SIZE(k) + r > c->sb.bucket_size)
			return "bad, length too big";
		if (ca && bucket <  ca->sb.first_bucket)
			return "bad, short offset";
		if (ca && bucket >= ca->sb.nbuckets)
			return "bad, offset past end of device";
		if (ca && ptr_stale(c, k, i))
			return "stale";
	}

	if (!bkey_cmp(k, &ZERO_KEY))
		return "bad, null key";
	if (!KEY_PTRS(k))
		return "bad, no pointers";
	if (!KEY_SIZE(k))
		return "zeroed key";
	return "";
}

static bool __ptr_invalid(struct cache_set *c, int level, const struct bkey *k)
{
	if (level && (!KEY_PTRS(k) || !KEY_SIZE(k)))
		goto bad;

	if (!KEY_SIZE(k))
		return true;

	for (unsigned i = 0; i < KEY_PTRS(k); i++) {
		struct cache *ca = PTR_CACHE(c, k, i);
		size_t bucket = PTR_BUCKET_NR(c, k, i);
		size_t r = PTR_OFFSET(k, i) & ~(~0 << c->bucket_bits);

		if (KEY_SIZE(k) + r > c->sb.bucket_size ||
		    PTR_DEV(k, i) > MAX_CACHES_PER_SET)
			goto bad;

		if (ca &&
		    (bucket <  ca->sb.first_bucket ||
		     bucket >= ca->sb.nbuckets))
			goto bad;
	}

	return false;
bad:
	cache_bug(c, "spotted bad key %s: %s", pkey(k), ptr_status(c, k));
	return true;
}

static bool ptr_invalid(struct btree *b, const struct bkey *k)
{
	return __ptr_invalid(b->c, b->level, k);
}

static bool ptr_bad(struct btree *b, const struct bkey *k)
{
	struct bucket *g;
	const char *err;
	unsigned i, stale;

	if (!bkey_cmp(k, &ZERO_KEY) || !KEY_PTRS(k) || ptr_invalid(b, k))
		return true;

	for (i = 0; i < KEY_PTRS(k); i++) {
		if (!PTR_CACHE(b->c, k, i))
			return true;

		g = PTR_BUCKET(b->c, k, i);
		stale = ptr_stale(b->c, k, i);

		cache_bug_on(stale > 96, b, "key too stale: %i, need_gc %u",
			     stale, b->c->need_gc);

		cache_bug_on(stale && KEY_DIRTY(k) && KEY_SIZE(k),
			     b, "stale dirty pointer");

		if (stale)
			return true;

		if (b->level) {
			err = "btree";
			if (KEY_DIRTY(k) ||
			    g->prio != btree_prio ||
			    g->heap != -1)
				goto bug;
		} else {
			err = "data";
			if (g->prio == btree_prio)
				goto bug;

			err = "dirty";
			if (KEY_DIRTY(k) && g->heap != -1)
				goto bug;
		}
	}

	return false;
bug:
	cache_bug(b, "inconsistent %s pointer %s: bucket %li heap %li pin %i "
		  "prio %i gen %i last_gc %i mark %i gc_gen %i", err, pkey(k),
		  PTR_BUCKET_NR(b->c, k, i), g->heap, atomic_read(&g->pin),
		  g->prio, g->gen, g->last_gc, g->mark, g->gc_gen);
	return true;
}

static struct bkey *next_recurse_key(struct btree *b, struct bkey *search)
{
	struct bkey *k, *ret = NULL;

	for_each_key_after_filter(b, k, search, ptr_bad) {
		if (!ret || bkey_cmp(k, ret) < 0)
			ret = k;
		/* We're actually in two loops here, looping over the sorted
		 * sets and then the keys within each set - break out of the
		 * inner loop and still loop over the sorted sets
		 */
		break;
	}

	return ret;
}

static bool should_split(struct btree *b)
{
	struct bset *i = write_block(b);
	return b->written >= btree_blocks(b) ||
		(i->seq == b->data->seq &&
		 b->written + __set_blocks(i, i->keys + 15, b->c)
		 > btree_blocks(b));
}

/* Debug code */

static void vdump_bucket_and_panic(struct btree *b, const char *m, va_list args)
{
	struct bkey *k;

	acquire_console_sem();

	for_each_key(b, k) {
		printk(KERN_ERR "block %i key %zu/%i: %s", index(*_i, b),
		       (uint64_t *) k - (*_i)->d, (*_i)->keys, pkey(k));

		for (unsigned i = 0; i < KEY_PTRS(k); i++) {
			size_t j = PTR_BUCKET_NR(b->c, k, i);
			printk(" bucket %li", j);

			if (j >= b->c->sb.first_bucket && j < b->c->sb.nbuckets)
				printk(" prio %i",
				       PTR_BUCKET(b->c, k, i)->prio);
		}

		printk(" %s\n", ptr_status(b->c, k));

		if (next(k) != end(*_i) &&
		    bkey_cmp(k, &START_KEY(next(k))) > 0)
			printk(KERN_ERR "Key skipped backwards\n");
	}

	vprintk(m, args);

	release_console_sem();

	panic("at %s\n", pbtree(b));
}

static void dump_bucket_and_panic(struct btree *b, const char *m, ...)
{
	va_list args;
	va_start(args, m);
	vdump_bucket_and_panic(b, m, args);
	va_end(args);
}

static void __maybe_unused
dump_key_and_panic(struct btree *b, struct bset *i, int j)
{
	long bucket = PTR_BUCKET_NR(b->c, node(i, j), 0);
	long r = PTR_OFFSET(node(i, j), 0) & ~(~0 << b->c->bucket_bits);

	printk(KERN_ERR "level %i block %i key %i/%i: %s "
	       "bucket %llu offset %li into bucket\n",
	       b->level, index(i, b), j, i->keys, pkey(node(i, j)),
	       (uint64_t) bucket, r);
	dump_bucket_and_panic(b, "");
}

#ifdef CONFIG_BCACHE_EDEBUG

static unsigned count_data(struct btree *b)
{
	unsigned ret = 0;
	struct bkey *k;

	if (!b->level)
		for_each_key(b, k)
			if (!ptr_invalid(b, k))
				ret += KEY_SIZE(k);
	return ret;
}

#define DUMP_BUCKET_BUG_ON(condition, b, ...) do {	\
	if (condition)					\
		dump_bucket_and_panic(b, __VA_ARGS__);	\
} while (0)

static void check_key_order_msg(struct btree *b, struct bset *i,
				const char *m, ...)
{
	if (!b->level && i->keys)
		for (struct bkey *k = i->start; next(k) < end(i); k = next(k))
			if (bkey_cmp(k, &START_KEY(next(k))) > 0) {
				va_list args;
				va_start(args, m);

				vdump_bucket_and_panic(b, m, args);
				va_end(args);
			}
}

#define check_key_order(b, i)	check_key_order_msg(b, i, "keys out of order")

#else /* EDEBUG */

#define count_data(b)					0
#define DUMP_BUCKET_BUG_ON(condition, b, ...)		BUG_ON(condition)
#define check_key_order(b, i)				do {} while (0)
#define check_key_order_msg(b, i, ...)			do {} while (0)

#endif

/* Btree IO */

static void btree_bio_resubmit(struct work_struct *w)
{
	struct btree *b = container_of(to_delayed_work(w), struct btree, work);
	bio_submit_split(&b->bio, &b->io, b->c->bio_split);
}

static void btree_bio_init(struct btree *b)
{
	bio_reset(&b->bio);
	b->bio.bi_sector   = PTR_OFFSET(&b->key, 0) +
		b->written * b->c->sb.block_size;
	b->bio.bi_bdev	   = PTR_CACHE(b->c, &b->key, 0)->bdev;
	b->bio.bi_rw	   = REQ_META;
}

static void fill_bucket_work(struct work_struct *w)
{
	struct btree *b = container_of(to_delayed_work(w), struct btree, work);
	struct bset *i = b->data;
	struct btree_iter *iter = b->c->fill_iter;
	const char *err = "bad btree header";
	BUG_ON(b->nsets || b->written);

	mutex_lock(&b->c->fill_lock);
	iter->top = iter->sets;

	if (!b->data->seq)
		goto err;

	for (i = b->data;
	     b->written < btree_blocks(b) && i->seq == b->data->seq;
	     i = write_block(b)) {
		err = "bad btree header";
		if (b->written + set_blocks(i, b->c) > btree_blocks(b))
			goto err;

		err = "bad magic";
		if (i->magic != bset_magic(b->c))
			goto err;

		err = "bad checksum";
		if (i->csum != csum_set(i))
			goto err;

		err = "empty set";
		if (i != b->data && !i->keys)
			goto err;

		if (i->keys) {
			cache_bug_on(bkey_cmp(&b->key, last_key(i)) < 0,
				     b, "short btree key");

			iter->top->k	= i->start;
			iter->top->end	= end(i);

			btree_iter_bubble(iter);
			iter->top++;
		}

		b->written += set_blocks(i, b->c);
	}

	err = "corrupted btree";
	for (i = write_block(b);
	     index(i, b) < btree_blocks(b);
	     i = ((void *) i) + block_bytes(b->c))
		if (i->seq == b->data->seq)
			goto err;

	iter->top--;
	__btree_sort(b, 0, NULL, iter, !b->level);

	pr_latency(b->expires, "fill_bucket");

	smp_wmb(); /* b->nread is our write lock */
	atomic_set(&b->nread, 1);

	if (0) {
err:		atomic_set(&b->nread, -1);
		cache_bug(b, "%s at bucket %lu, block %i, %i keys",
			  err, PTR_BUCKET_NR(b->c, &b->key, 0),
			  index(i, b), i->keys);
	}

	mutex_unlock(&b->c->fill_lock);

	atomic_set(&b->io, -1);
	closure_run_wait(&b->wait, delayed);
}

static void fill_bucket_endio(struct bio *bio, int error)
{
	struct btree *b = bio->bi_private;
	bio_put(bio);

	if (error) {
		cache_set_error(b->c, "reading index");
		atomic_set(&b->nread, -1);
	}

	if (!atomic_dec_and_test(&b->io))
		return;

	PREPARE_DELAYED_WORK(&b->work, fill_bucket_work);

	if (atomic_read(&b->nread) == -1) {
		atomic_set(&b->io, -1);
		closure_run_wait(&b->wait, delayed);
	} else
		BUG_ON(!schedule_work(&b->work.work));
}

static void fill_bucket(struct btree *b)
{
	BUG_ON(b->nsets || b->written);
	BUG_ON(atomic_xchg(&b->io, 1) != -1);

	cancel_delayed_work_sync(&b->work);
	b->expires = jiffies;

	btree_bio_init(b);
	b->bio.bi_rw	       |= READ_SYNC;
	b->bio.bi_size		= KEY_SIZE(&b->key) << 9;
	b->bio.bi_end_io	= fill_bucket_endio;
	b->bio.bi_private	= b;

	bio_map(&b->bio, b->data);

	if (bio_submit_split(&b->bio, &b->io, b->c->bio_split)) {
		PREPARE_DELAYED_WORK(&b->work, btree_bio_resubmit);
		BUG_ON(!schedule_work(&b->work.work));
	}
	pr_debug("%s", pbtree(b));
}

static void btree_write_endio(struct bio *bio, int error)
{
	int n;
	struct bio_vec *bv;
	struct btree_write *w = bio->bi_private;
	struct btree *b = w->b;
	bio_put(bio);

	cache_set_err_on(error, b->c, "writing index");

	if (!atomic_dec_and_test(&b->io))
		return;

	pr_latency(w->wait_time, "btree write");

	if (!w->nofree)
		__bio_for_each_segment(bv, &b->bio, n, 0)
			__free_page(bv->bv_page);

	closure_run_wait(&w->wait, delayed);
	if (w->owner)
		closure_put(w->owner, delayed);

	if (w->prio_blocked &&
	    !atomic_sub_return(w->prio_blocked, &b->c->prio_blocked))
		closure_run_wait(&b->c->bucket_wait, delayed);

	if (w->journal) {
		atomic_dec_bug(w->journal);
		w->journal = NULL;
		closure_run_wait(&b->c->journal.wait, delayed);
		if (journal_full(b->c))
			schedule_work(&b->c->journal.work);
	}

	memset(w, 0, sizeof(struct btree_write));
	atomic_set(&b->io, -1);
	closure_run_wait(&b->wait, delayed);

	if (b->write) {
		long delay = max_t(long, 0, b->expires - jiffies);
		schedule_delayed_work(&b->work, delay);
	}
}

static int __btree_write(struct btree *b)
{
	int j;
	struct bio_vec *bv;
	struct btree_write *w;
	struct bset *i = write_block(b);
	void *base = (void *) ((unsigned long) i & ~(PAGE_SIZE - 1));

	if (atomic_cmpxchg(&b->io, -1, 1) != -1)
		return -1;

	/* XXX: get rid of this since we have b->io? */
	w = xchg(&b->write, NULL);
	if (!w) {
		/* We raced, first saw b->write before the write was
		 * started, but the write has already completed.
		 */
		atomic_set(&b->io, -1);
		return -1;
	}

	__cancel_delayed_work(&b->work);
	pr_latency(w->wait_time, "btree write");
	set_wait(w);

	BUG_ON(b->written && !i->keys);
	check_key_order(b, i);
	i->csum = csum_set(i);

	btree_bio_init(b);
	b->bio.bi_rw	       |= WRITE_SYNC;
	b->bio.bi_size		= set_blocks(i, b->c) * block_bytes(b->c);
	b->bio.bi_end_io	= btree_write_endio;
	b->bio.bi_private	= w;

	bio_map(&b->bio, i);
	if (bio_alloc_pages(&b->bio, GFP_NOIO))
		goto err;

	bio_for_each_segment(bv, &b->bio, j)
		memcpy(page_address(bv->bv_page),
		       base + j * PAGE_SIZE, PAGE_SIZE);

	if (bio_submit_split(&b->bio, &b->io, b->c->bio_split)) {
		cancel_delayed_work_sync(&b->work);
		PREPARE_DELAYED_WORK(&b->work, btree_bio_resubmit);
		BUG_ON(!schedule_work(&b->work.work));
	}

	if (0) {
		struct closure wait;
err:		closure_init_stack(&wait);

		if (current->bio_list) {
			atomic_set(&b->io, -1);
			b->write = w;
			return -1;
		}

		w->nofree = true;
		bio_map(&b->bio, i);

		BUG_ON(!closure_wait(&w->wait, &wait));
		bio_submit_split(&b->bio, &b->io, b->c->bio_split);
		closure_sync(&wait);
	}

	if (b->written) {
		atomic_long_inc(&b->c->btree_write_count);
		atomic_long_add(i->keys, &b->c->keys_write_count);
	}

	pr_debug("%s block %i keys %i", pbtree(b), b->written, i->keys);

	b->written += set_blocks(i, b->c);
	atomic_long_add(set_blocks(i, b->c) * b->c->sb.block_size,
			&PTR_CACHE(b->c, &b->key, 0)->btree_sectors_written);
	return 0;
}

static void btree_write_work(struct work_struct *w)
{
	struct btree *b = container_of(to_delayed_work(w), struct btree, work);

	pr_latency(b->expires, "btree_write_work");

	smp_mb(); /* between unlock/requeue from rw_unlock */
	if (down_read_trylock(&b->lock))
		rw_unlock(false, b);
}

static void btree_write(struct btree *b, bool now, struct search *s)
{
	struct bset *i = write_block(b);

	BUG_ON(!now && !s);
	BUG_ON(b->written &&
	       (b->written >= btree_blocks(b) ||
		i->seq != b->data->seq ||
		!i->keys));

	if (!b->write) {
		b->write = &b->writes[b->next];
		b->write->b = b;
		b->write->journal = NULL;
		b->next ^= 1;

		PREPARE_DELAYED_WORK(&b->work, btree_write_work);
		b->expires = jiffies + msecs_to_jiffies(30000);
	}

	b->write->prio_blocked += b->prio_blocked;
	b->prio_blocked = 0;

	if (s && s->journal && !b->level) {
		if (b->write->journal &&
		    journal_pin_cmp(b->c, b->write, s)) {
			atomic_dec_bug(b->write->journal);
			b->write->journal = NULL;
		}

		if (!b->write->journal) {
			b->write->journal = s->journal;
			atomic_inc(b->write->journal);
		}
	}


	/* Force write if set is too big */
	if (!now &&
	    !b->level &&
	    (set_bytes(i) < 8000 ||
	     set_blocks(i, b->c) == __set_blocks(i, i->keys + 15, b->c)))
		return;

#ifdef DEBUG_LATENCY
	if (!b->write->wait_time)
		set_wait(b->write);
#endif
	if (s && now) {
		/* Must wait on multiple writes */
		BUG_ON(b->write->owner);
		b->write->owner = &s->insert;
		closure_get(&s->insert);
	}

	if (__btree_write(b)) {
		b->expires = jiffies;
		if (b->work.timer.function)
			mod_timer_pending(&b->work.timer, b->expires);
	}

	BUG_ON(!b->written);
}

/* Btree cache */

static void free_bucket(struct btree *b)
{
	lockdep_assert_held(&b->c->bucket_lock);
	BUG_ON(b->write);

	if (b->data)
		list_move_tail(&b->lru, &b->c->lru);
	else
		list_move_tail(&b->lru, &b->c->freed);

	b->key.ptr[0] = 0;
	b->written = 0;
	b->nsets = 0;
	atomic_set(&b->nread, 0);
	__cancel_delayed_work(&b->work);

	hlist_del_init_rcu(&b->hash);
}

static int reap_bucket(struct btree *b, struct closure *cl)
{
	lockdep_assert_held(&b->c->bucket_lock);

	if (!down_write_trylock(&b->lock))
		return -1;

	BUG_ON(!b->data);
	if (b->write || atomic_read(&b->io) != -1) {
		if (b->write && time_is_after_jiffies(b->expires))
			b->expires = jiffies;

		if (b->write && cl) {
			spin_unlock(&b->c->bucket_lock);
			__btree_write(b);
			spin_lock(&b->c->bucket_lock);
		}

		rw_unlock_nowrite(true, b);

		if (!cl)
			return -1;

		closure_wait_on_async(&b->wait, delayed, cl,
				      !b->write && atomic_read(&b->io) == -1);
		return -EAGAIN;
	}

	return 0;
}

static int shrink_buckets(int nr, gfp_t flags)
{
	struct btree *oldest_bucket(struct cache_set *c)
	{
		struct btree *ret = NULL, *b;
		list_for_each_entry(b, &c->lru, lru)
			if (!ret || time_after(ret->jiffies, b->jiffies))
				ret = b;
		return ret;
	}

	struct cache_set *c;
	struct btree *b;
	int ret = 0, reserve, orig;

	if (list_empty(&cache_sets))
		return 0;

	c = list_first_entry(&cache_sets, struct cache_set, list);

	spin_lock(&c->bucket_lock);

	orig = nr /= c->btree_pages;
	reserve = btree_reserve(c);

	list_for_each_entry(b, &c->lru, lru)
		ret++;

	ret = max(ret - reserve, 0);

	while (nr && ret && !c->try_harder) {
		b = oldest_bucket(c);
		if (reap_bucket(b, NULL))
			break;

		free_pages((unsigned long) b->data, b->page_order);
		b->data = NULL;
		free_bucket(b);
		rw_unlock(true, b);
		nr--, ret--;
	}

	spin_unlock(&c->bucket_lock);

	if (orig)
		pr_debug("wanted %i freed %i now %i", orig, orig - nr, ret);
	return ret * c->btree_pages;
}

static int btree_cache_size(struct cache_set *c)
{
	struct list_head *l;
	int i = 0;

	spin_lock(&c->bucket_lock);
	list_for_each(l, &c->lru)
		i++;
	spin_unlock(&c->bucket_lock);

	return i;
}

static struct hlist_head *hash_bucket(struct cache_set *c, struct bkey *k)
{
	return &c->bucket_hash[hash_64(PTR_HASH(k), BUCKET_HASH_BITS)];
}

static struct btree *find_bucket(struct cache_set *c, struct bkey *k)
{
	struct hlist_node *cursor;
	struct btree *b;

	rcu_read_lock();
	hlist_for_each_entry_rcu(b, cursor, hash_bucket(c, k), hash)
		if (PTR_HASH(&b->key) == PTR_HASH(k))
			goto out;
	b = NULL;
out:
	rcu_read_unlock();
	return b;
}

static void alloc_bucket_data(struct btree *b)
{
	int pages = KEY_SIZE(&b->key) / PAGE_SECTORS;
	b->page_order = ilog2(max(b->c->btree_pages, pages));
	b->data = (void *) __get_free_pages(__GFP_NOWARN|GFP_NOIO,
					      b->page_order);
}

static struct btree *__alloc_bucket(struct cache_set *c, gfp_t flags)
{
	struct btree *b = kzalloc(sizeof(*b) + sizeof(struct bio_vec) *
				  bucket_pages(c), flags);

	if (b) {
		INIT_LIST_HEAD(&b->lru);
		init_rwsem(&b->lock);
		INIT_DELAYED_WORK(&b->work, NULL);
		b->c = c;
		atomic_set(&b->io, -1);
		b->bio.bi_max_vecs	= bucket_pages(b->c);
		b->bio.bi_io_vec	= b->bio.bi_inline_vecs;
	}
	return b;
}

/* Caller must have locked bucket_lock; always returns with bucket_lock
 * unlocked
 */
static struct btree *alloc_bucket(struct cache_set *c, struct bkey *k,
				  struct closure *cl)
{
	struct btree *init_bucket(struct btree *b)
	{
		if (!find_bucket(c, k)) {
			BUG_ON(atomic_read(&b->io) != -1);

			bkey_copy(&b->key, k);
			list_move(&b->lru, &c->lru);
			hlist_del_init_rcu(&b->hash);
			hlist_add_head_rcu(&b->hash, hash_bucket(c, k));
		} else {
			up_write(&b->lock);
			b = NULL;
		}
		spin_unlock(&c->bucket_lock);
		return b;
	}

	struct btree *b, *i;
	int pages = KEY_SIZE(k) / PAGE_SECTORS;

	lockdep_assert_held(&c->bucket_lock);
	BUG_ON(list_empty(&c->lru));

	b = list_entry(c->lru.prev, struct btree, lru);
	if (pages <= c->btree_pages &&
	    !PTR_HASH(&b->key) &&
	    !reap_bucket(b, NULL))
		return init_bucket(b);

	list_for_each_entry(b, &c->freed, lru)
		if (atomic_read(&b->io) == -1 &&
		    !work_pending(&b->work.work) &&
		    down_write_trylock(&b->lock)) {
			BUG_ON(b->data);
			goto out;
		}

	spin_unlock(&c->bucket_lock);

	b = __alloc_bucket(c, GFP_NOIO);
	if (!b)
		goto err;

	BUG_ON(!down_write_trylock(&b->lock));

	spin_lock(&c->bucket_lock);
out:
	if (!init_bucket(b))
		return NULL;

	alloc_bucket_data(b);
	if (!b->data)
		goto err;

	return b;
err:
	spin_lock(&c->bucket_lock);

	if (b) {
		free_bucket(b);
		rw_unlock(true, b);
	}
retry:
	b = ERR_PTR(-ENOMEM);

	if (pages > c->btree_pages || !cl) {
		spin_unlock(&c->bucket_lock);
		return b;
	}

	if (!c->try_harder || c->try_harder == cl) {
		/* XXX: tracepoint */
		c->try_harder = cl;

		list_for_each_entry_reverse(i, &c->lru, lru) {
			int e = reap_bucket(i, cl);
			if (e == -EAGAIN)
				b = ERR_PTR(-EAGAIN);
			if (!e)
				return init_bucket(i);
		}

		if (b == ERR_PTR(-EAGAIN) &&
		    test_bit(CLOSURE_BLOCK, &cl->flags)) {
			spin_unlock(&c->bucket_lock);
			closure_sync(cl);
			spin_lock(&c->bucket_lock);
			goto retry;
		}
	} else {
		closure_wait_on_async(&c->try_wait, delayed,
				      cl, !c->try_harder);
		b = ERR_PTR(-EAGAIN);
	}

	spin_unlock(&c->bucket_lock);
	return b;
}

static struct btree *get_bucket(struct cache_set *c, struct bkey *k,
				int level, bool write, struct closure *s)
{
	int nread;
	struct btree *b;
	BUG_ON(level < 0);
retry:
	b = find_bucket(c, k);

	if (!b) {
		spin_lock(&c->bucket_lock);
		b = alloc_bucket(c, k, s);
		if (!b)
			goto retry;
		if (IS_ERR(b))
			return b;

		atomic_set(&b->nread, 0);
		b->level	= level;
		b->written	= 0;
		b->nsets	= 0;
		lock_set_subclass(&b->lock.dep_map, level + 1, _THIS_IP_);

		fill_bucket(b);

		if (!write)
			downgrade_write(&b->lock);
	} else {
		rw_lock(write, b, level);
		if (PTR_HASH(&b->key) != PTR_HASH(k)) {
			rw_unlock(write, b);
			goto retry;
		}
		BUG_ON(b->level != level);
	}

	b->jiffies = jiffies;

	nread = closure_wait_on(&b->wait, delayed, s, atomic_read(&b->nread));
	if (nread != 1) {
		rw_unlock(write, b);
		b = ERR_PTR(nread ? -EIO : -EAGAIN);
	} else
		BUG_ON(!b->written);

	return b;
}

#define insert_lock(s, b)	((b)->level	<= (s)->lock)

#define btree(f, b, k, s, ...)						\
({									\
	int _r, l = b->level - 1;					\
	bool _w = l <= (s)->lock;					\
	struct btree *_b = get_bucket(b->c, k, l, _w, &(s)->insert);	\
	BUG_ON(ptr_bad(b, k));						\
	if (!IS_ERR(_b)) {						\
		_r = btree_ ## f(_b, s, ## __VA_ARGS__);		\
		rw_unlock(_w, _b);					\
	} else								\
		_r = PTR_ERR(_b);					\
	_r;								\
})

#define btree_root(f, c, s, ...)					\
({									\
	int _r = -EINTR;						\
	do {								\
		struct btree *_b = (c)->root;				\
		bool _w = insert_lock(s, _b);				\
		rw_lock(_w, _b, _b->level);				\
		if (_b == (c)->root &&					\
		    _w == insert_lock(s, _b))				\
			_r = btree_ ## f(_b, s, ## __VA_ARGS__);	\
		rw_unlock(_w, _b);					\
	} while (_r == -EINTR);						\
									\
	if ((c)->try_harder == &(s)->insert) {				\
		(c)->try_harder = NULL;					\
		closure_run_wait(&(c)->try_wait, delayed);		\
	}								\
	_r;								\
})

/* Btree alloc */

static void btree_free(struct btree *b, struct search *s)
{
	/* The BUG_ON() in get_bucket() implies that we must have a write lock
	 * on parent to free or even invalidate a node
	 */
	BUG_ON(s->lock <= b->level);
	BUG_ON(b == b->c->root);
	pr_debug("bucket %s", pbtree(b));

	spin_lock(&b->c->bucket_lock);

	for (unsigned i = 0; i < KEY_PTRS(&b->key); i++) {
		BUG_ON(atomic_read(&PTR_BUCKET(b->c, &b->key, i)->pin));

		inc_gen(PTR_CACHE(b->c, &b->key, i),
			PTR_BUCKET(b->c, &b->key, i));
	}

	/* This isn't correct, the caller needs to add the wait list
	 * to the wait list for the new bucket's write.
	 */
	if (b->write) {
		BUG_ON(b->write->owner);
		BUG_ON(b->write->prio_blocked);
		closure_run_wait(&b->write->wait, delayed);
		if (b->write->journal)
			atomic_dec_bug(b->write->journal);
		b->write->journal = NULL;
		b->write = NULL;
	}

	unpop_bucket(b->c, &b->key);
	free_bucket(b);
	spin_unlock(&b->c->bucket_lock);
}

static struct btree *btree_alloc(struct cache_set *c, int level,
				 struct closure *cl)
{
	BKEY_PADDED(key) k;
	struct btree *b = ERR_PTR(-EAGAIN);
retry:
	spin_lock(&c->bucket_lock);
	if (__pop_bucket_set(c, btree_prio, &k.key, 1, cl))
		goto err_unlock;

	SET_KEY_SIZE(&k.key, c->btree_pages * PAGE_SECTORS);
retry_alloc:
	b = alloc_bucket(c, &k.key, cl);
	if (IS_ERR(b))
		goto err;

	/* A btree pointer may occasionally be invalidated without btree_free()
	 * being called, thus the bucket may potentially be cached while
	 * legitimately free.
	 */
	if (!b) {
		b = find_bucket(c, &k.key);
		/* this is bothersome - but it's probably a harmless race
		 * with gc
		 * XXX: might not be a bad idea to trace this stuff
		 */
		if (!down_write_trylock(&b->lock))
			goto retry;

		if (PTR_HASH(&b->key) != PTR_HASH(&k.key)) {
			/* belt and suspenders */
			rw_unlock(true, b);
			spin_lock(&c->bucket_lock);
			goto retry_alloc;
		}
	}

	b->jiffies	= jiffies;
	atomic_set(&b->nread, 1);
	b->level	= level;
	b->written	= 0;
	b->nsets	= 0;
	lock_set_subclass(&b->lock.dep_map, level + 1, _THIS_IP_);

	bset_init(b, b->data);

	return b;
err:
	spin_lock(&c->bucket_lock);

	unpop_bucket(c, &k.key);
	__bkey_put(c, &k.key);
err_unlock:
	spin_unlock(&c->bucket_lock);
	return b;
}

/* Cache lookup */

static struct bio *cache_hit(struct btree *b, struct bio *bio,
			     struct bkey *k, struct search *s)
{
	sector_t sector = bio->bi_sector;
	unsigned sectors = k->key - sector;
	struct bio *ret;
	struct block_device *bdev;
	struct bucket *g = PTR_BUCKET(b->c, k, 0);

	if (keylist_realloc(&s->keys, 1))
		return ERR_PTR(-ENOMEM);

	atomic_inc(&g->pin);
	smp_mb__after_atomic_inc();

	if (ptr_stale(b->c, k, 0)) {
		atomic_dec_bug(&g->pin);
		return NULL;
	}

	bdev = PTR_CACHE(b->c, k, 0)->bdev;
	sector += KEY_SIZE(k) - k->key + PTR_OFFSET(k, 0);
	sectors = min(sectors, __bio_max_sectors(bio, bdev, sector));

	ret = bio_split_c(bio, sectors, s->d->c);
	if (!ret) {
		atomic_dec_bug(&g->pin);
		return ERR_PTR(-ENOMEM);
	}

	ret->bi_sector	= sector;
	ret->bi_bdev	= bdev;
	ret->bi_end_io	= cache_request_endio;

	if (ret != bio) {
		closure_get(&s->cl);
		ret->bi_rw &= ~REQ_UNPLUG;
	}

	/* For multiple cache devices, copy only the pointer we're actually
	 * reading from
	 */
	bkey_copy(s->keys.top, k);
	SET_KEY_PTRS(s->keys.top, 1);
	keylist_push(&s->keys);

	g->prio = initial_prio;
			/* * (cache_hit_seek + cache_hit_priority
			 * bio_sectors(bio) / c->sb.bucket_size)
			/ (cache_hit_seek + cache_hit_priority);*/

	pr_debug("cache hit of %i sectors from %llu, need %i sectors",
		 bio_sectors(ret), (uint64_t) ret->bi_sector,
		 ret == bio ? 0 : bio_sectors(bio));

	return ret;
}

#define SEARCH(s) KEY(s->d->id, s->bio->bi_sector, 0)

static int btree_search(struct btree *b, struct search *s, uint64_t *reada)
{
	struct bio *n;
	struct btree_iter iter;
	btree_iter_init(b, &iter, &SEARCH(s), 0);

	while (1) {
		struct bkey *k = btree_iter_next(&iter);
		if (!k || KEY_DEV(k) != s->d->id)
			return 0;

		if (ptr_bad(b, k))
			continue;

		if (bio_end(s->bio) <= KEY_START(k)) {
			*reada = min(*reada, KEY_START(k));
			return 0;
		}

		while (s->bio->bi_sector < KEY_START(k)) {
			int sectors = min_t(int, bio_max_sectors(s->bio),
					    KEY_START(k) - s->bio->bi_sector);

			n = bio_split_c(s->bio, sectors, s->d->c);
			if (!n)
				return -ENOMEM;

			BUG_ON(n == s->bio);
			closure_get(&s->cl);
			generic_make_request(n);
		}

		pr_debug("%s", pkey(k));

		do {
			n = cache_hit(b, s->bio, k, s);
			if (!n)
				break;
			if (IS_ERR(n))
				return -ENOMEM;

			generic_make_request(n);

			if (n == s->bio) {
				s->cache_hit = true;
				return 0;
			}
		} while (s->bio->bi_sector < k->key);
	}
}

static int btree_search_recurse(struct btree *b, struct search *s,
				uint64_t *reada)
{
	int ret = -1;
	struct bkey search = SEARCH(s), *k = &search;

	pr_debug("at %s searching for %llu", pbtree(b), search.key);

	if (!b->level)
		return btree_search(b, s, reada);

	while ((k = next_recurse_key(b, k))) {
		ret = btree(search_recurse, b, k, s, reada);

		if (ret ||
		    s->cache_hit ||
		    bkey_cmp(k, &KEY(s->d->id, bio_end(s->bio), 0)) >= 0)
			return ret;
	}

	cache_bug_on(ret == -1, b, "no key to recurse on at level %i/%i",
		     b->level, b->c->root->level);
	return 0;
}

/* Garbage collection */

static bool btree_try_merge(struct btree *b, struct bkey *l, struct bkey *r)
{
	if (KEY_PTRS(l) != KEY_PTRS(r) ||
	    KEY_DIRTY(l) != KEY_DIRTY(r) ||
	    bkey_cmp(l, &START_KEY(r)))
		return false;

	for (unsigned j = 0; j < KEY_PTRS(l); j++)
		if (l->ptr[j] + PTR(0, KEY_SIZE(l), 0) != r->ptr[j] ||
		    PTR_BUCKET(b->c, l, j) != PTR_BUCKET(b->c, r, j))
			return false;

	SET_KEY_SIZE(l, KEY_SIZE(l) + KEY_SIZE(r));
	l->key		+= KEY_SIZE(r);
	bkey_copy(r, l);
	return true;
}

static void __btree_sort(struct btree *b, int start, struct bset *new,
			 struct btree_iter *iter, bool fixup)
{
	void do_fixups(void)
	{
		bool overlap(struct bkey *l, struct bkey *r)
		{
			return bkey_cmp(l, &START_KEY(r)) > 0;
		}

		void cut_front_set(struct bkey *where, struct btree_iter_set *i)
		{
			struct bkey *k = i->k;

			do {
				__cut_front(where, k);
				k = next(k);
			} while (k != i->end && overlap(where, k));

			__btree_iter_bubble(iter, i);
		}

		struct btree_iter_set *top = iter->top, *i = top - 1;

		if (iter->top <= iter->sets)
			return;

		while (bkey_cmp(&START_KEY(top->k), &START_KEY(i->k)) == 0) {
			if (top->k < i->k)
				swap(*top, *i);

			if (!KEY_SIZE(top->k))
				return;

			cut_front_set(top->k, i);
		}

		while (i >= iter->sets && overlap(top->k, i->k)) {
			if (top->k < i->k) {
				struct bkey *k = i->k;

				do {
					if (KEY_SIZE(k))
						cut_back(&START_KEY(k), top->k);

					k = next(k);
				} while (k != i->end && overlap(top->k, k));

				--i;
			} else
				cut_front_set(top->k, i);
		}
	}

	bool bad(struct bkey *k)
	{
		return b->written && !new
			? ptr_invalid(b, k)
			: ptr_bad(b, k);
	}

	size_t oldsize = 0, order = b->page_order, keys = 0;
	struct bset *out = new;
	struct bkey *k, *last = NULL;

	if (!fixup)
		oldsize = count_data(b);

	if (start) {
		struct bset *i;
		for_each_sorted_set_start(b, i, start)
			keys += i->keys;

		order = roundup_pow_of_two(__set_bytes(i, keys)) / PAGE_SIZE;
		if (order)
			order = ilog2(order);
	}

	if (!out)
		out = (void *) __get_free_pages(__GFP_NOWARN|GFP_NOIO, order);
	if (!out) {
		mutex_lock(&b->c->sort_lock);
		out = b->c->sort;
		order = ilog2(bucket_pages(b->c));
	}

	while (!btree_iter_end(iter)) {
		if (fixup)
			do_fixups();

		k = btree_iter_next(iter);

		if (!bad(k)) {
			if (!last)
				last = out->start;
			else if (b->level ||
				 !btree_try_merge(b, last, k))
				last = next(last);

			bkey_copy(last, k);
		}
	}

	out->keys = last ? (uint64_t *) next(last) - out->d : 0;

	if (new)
		return;

	b->nsets = start;

	if (!start && order == b->page_order) {
		out->magic	= bset_magic(b->c);
		out->seq	= b->data->seq;
		out->version	= b->data->version;
		swap(out, b->data);

		if (b->c->sort == b->data)
			b->c->sort = out;
	} else {
		b->sets[start]->keys = out->keys;
		memcpy(b->sets[start]->start, out->start,
		       (void *) end(out) - (void *) out->start);
	}

	if (out == b->c->sort)
		mutex_unlock(&b->c->sort_lock);
	else
		free_pages((unsigned long) out, order);

	pr_debug("sorted %i keys", b->sets[start]->keys);
	check_key_order(b, b->sets[start]);
	BUG_ON(!fixup && b->written && count_data(b) < oldsize);
}

static void btree_sort(struct btree *b, int start, struct bset *new)
{
	struct btree_iter iter;
	btree_iter_init(b, &iter, NULL, start);

	__btree_sort(b, start, new, &iter, false);
}

static void btree_mark_meta(struct cache_set *c)
{
	void mark_key(struct bkey *k)
	{
		for (unsigned i = 0; i < KEY_PTRS(k); i++)
			PTR_BUCKET(c, k, i)->mark = GC_MARK_BTREE;
	}

	struct cache *ca;
	uint64_t *i;

	if (c->root)
		mark_key(&c->root->key);

	mark_key(&c->uuid_bucket);

	for_each_cache(ca, c) {
		for (i = ca->sb.d; i < ca->sb.d + ca->sb.keys; i++)
			ca->buckets[*i].mark = GC_MARK_BTREE;

		for (i = ca->prio_buckets;
		     i < ca->prio_buckets + prio_buckets(ca) * 2; i++)
			ca->buckets[*i].mark = GC_MARK_BTREE;
	}
}

static void __btree_mark_key(struct cache_set *c, int level, struct bkey *k)
{
	if (!k->key)
		return;

	for (unsigned i = 0; i < KEY_PTRS(k); i++) {
		struct bucket *g = PTR_BUCKET(c, k, i);

		if (gen_after(g->gc_gen, PTR_GEN(k, i)))
			g->gc_gen = PTR_GEN(k, i);

		if (ptr_stale(c, k, i))
			continue;

		cache_bug_on(level
			     ? g->mark && g->mark != GC_MARK_BTREE
			     : g->mark < GC_MARK_DIRTY, c,
			     "inconsistent pointers: mark = %i, "
			     "level = %i", g->mark, level);

		if (level)
			g->mark = GC_MARK_BTREE;
		else if (KEY_DIRTY(k))
			g->mark = GC_MARK_DIRTY;
		else if (g->mark >= 0 &&
			 ((int) g->mark) + KEY_SIZE(k) < SHORT_MAX)
			g->mark += KEY_SIZE(k);
	}
}

#define btree_mark_key(b, k)	__btree_mark_key(b->c, b->level, k)

static int btree_gc_mark(struct btree *b, size_t *keys, struct gc_stat *gc)
{
	uint8_t ret = 0;
	struct bset *i;
	struct bkey *k;

	for_each_sorted_set(b, i)
		cache_bug_on(i->keys && bkey_cmp(&b->key, last_key(i)) < 0,
			     b, "found short btree key in gc");

	gc->nodes++;
	for_each_key_filter(b, k, ptr_bad) {
		*keys += 2 + KEY_PTRS(k);

		gc->key_bytes += 2 + KEY_PTRS(k);
		gc->nkeys++;

		gc->data += KEY_SIZE(k);
		if (KEY_DIRTY(k))
			gc->dirty += KEY_SIZE(k);
	}

	for_each_key_filter(b, k, ptr_invalid) {
		for (unsigned i = 0; i < KEY_PTRS(k); i++)
			ret = max(ret, ptr_stale(b->c, k, i));

		btree_mark_key(b, k);
	}

	return ret;
}

static int btree_gc_recurse(struct btree *b, struct search *s,
			    struct gc_stat *gc)
{
	struct btree *alloc(struct btree *r, struct bkey *k)
	{
		/* can't sleep in pop_bucket(), as we block priorities from
		 * being written
		 */
		struct btree *n = btree_alloc(r->c, r->level, NULL);

		if (!IS_ERR_OR_NULL(n)) {
			swap(r, n);
			btree_sort(n, 0, r->data);
			bkey_copy_key(&r->key, &n->key);

			memcpy(k->ptr, r->key.ptr,
			       sizeof(uint64_t) * KEY_PTRS(&r->key));

			__bkey_put(b->c, &r->key);
			atomic_inc(&b->c->prio_blocked);
			b->prio_blocked++;

			btree_free(n, s);
			rw_unlock(true, n);
		}

		return r;
	}

	void write(struct btree *r)
	{
		if (!r->written || r->write)
			btree_write(r, true, !b->written ? s : NULL);

		rw_unlock(true, r);
	}

	int ret = 0, stale;
	size_t dirty, keys, pkeys = 0;
	struct btree *r, *p = NULL;
	struct bkey *k;

	while ((k = next_recurse_key(b, &b->c->gc_done))) {
		r = get_bucket(b->c, k, b->level - 1, true, &s->insert);
		if (IS_ERR(r)) {
			ret = PTR_ERR(r);
			break;
		}

		keys = dirty = 0;
		stale = btree_gc_mark(r, &keys, gc);

		if (!b->written &&
		    (r->level || stale > 10))
			r = alloc(r, k);

		if (r->level)
			ret = btree_gc_recurse(r, s, gc);

		if (ret) {
			write(r);
			break;
		}

		bkey_copy_key(&b->c->gc_done, k);

		pkeys = __set_blocks(b->data, pkeys + keys, b->c);
		if (p && pkeys < (btree_blocks(b) * 2) / 3) {
			if (r->written)
				r = alloc(r, k);

			if (!r->written) {
				pr_debug("coalescing");
				r->nsets += p->nsets + 1;
				memcpy(&r->sets[1],
				       &p->sets[0],
				       sizeof(void *) * (p->nsets + 1));
				btree_sort(r, 0, NULL);

				btree_free(p, s);
				rw_unlock(true, p);

				p = NULL;
				keys = r->data->keys;
				gc->nodes--;
			}
		}

		if (p)
			write(p);

		lock_set_subclass(&r->lock.dep_map, 0, _THIS_IP_);
		p = r;
		pkeys = keys;
	}

	if (p)
		write(p);

	/* Might have freed some children, must remove their keys */
	btree_sort(b, 0, NULL);

	return ret;
}

static int btree_gc_root(struct btree *b, struct search *s, struct gc_stat *gc)
{
	struct btree *n = NULL;
	size_t keys = 0;
	int ret = 0, stale = btree_gc_mark(b, &keys, gc);

	if (b->level || stale > 10)
		n = btree_alloc(b->c, b->level, &s->insert);

	if (!IS_ERR_OR_NULL(n)) {
		swap(b, n);
		btree_sort(n, 0, b->data);
		bkey_copy_key(&b->key, &n->key);
	}

	if (b->level)
		ret = btree_gc_recurse(b, s, gc);

	if (!b->written || b->write) {
		atomic_inc(&b->c->prio_blocked);
		b->prio_blocked++;
		btree_write(b, true, n ? s : NULL);
	}

	if (!IS_ERR_OR_NULL(n)) {
		closure_sync(&s->insert);
		set_new_root(b);
		btree_free(n, s);
		rw_unlock(true, b);
	}

	return ret;
}

static void set_gc_sectors(struct cache_set *s)
{
	struct cache *c;
	uint64_t n = 0;
	for_each_cache(c, s)
		n += c->sb.nbuckets;

	atomic_set(&s->sectors_to_gc, s->sb.bucket_size * n / 8);
}

static unsigned btree_used(struct cache_set *c)
{
	uint64_t ret = c->gc_stats.key_bytes * 100;
	return ret / ((c->gc_stats.nodes ?: 1) * btree_bytes(c));
}

static void btree_gc(struct cache_set *s)
{
	unsigned long pinned = 0, time = jiffies;
	struct gc_stat stats;
	struct search sr;
	struct bucket *b;
	struct cache *c;
	uint8_t need_gc = 0;

	search_init_stack(&sr);
	sr.lock = SHORT_MAX;

	memcpy(&stats, &s->gc_stats, sizeof(struct gc_stat));
	stats.nodes = stats.key_bytes = stats.nkeys = 0;
	stats.data = stats.dirty = 0;

	spin_lock(&s->bucket_lock);
	for_each_cache(c, s)
		free_some_buckets(c);
	spin_unlock(&s->bucket_lock);

	if (!bkey_cmp(&s->gc_done, &KEY(0, 0, 0)))
		for_each_cache(c, s)
			for_each_bucket(b, c)
				if (!atomic_read(&b->pin))
					b->mark = 0;

	if (btree_root(gc_root, s, &sr, &stats)) {
		printk(KERN_WARNING "bcache: gc failed!\n");
		queue_work(delayed, &s->gc_work);
		goto out;
	}

	s->gc_done = KEY(0, 0, 0);
	set_gc_sectors(s);
	closure_sync(&sr.insert);

	/* Possibly wait for new UUIDs or whatever to hit disk */
	btree_journal_wait(s, &sr.insert);
	closure_sync(&sr.insert);

	spin_lock(&s->bucket_lock);
	swap(need_gc, s->need_gc);

	btree_mark_meta(s);

	s->min_prio = initial_prio;

	for_each_cache(c, s) {
		c->heap.size = 0;

		for_each_bucket(b, c) {
			cache_bug_on(gen_after(b->last_gc, b->gc_gen), c,
				     "found old gen in gc");

			b->heap		= -1;
			b->last_gc	= b->gc_gen;
			b->gc_gen	= b->gen;
			s->need_gc	= max(s->need_gc, bucket_gc_gen(b));

			if (b->prio)
				s->min_prio = min(s->min_prio, b->prio);

			if (!atomic_read(&b->pin))
				bucket_add_heap(c, b);
			else
				pinned++;
		}
	}

	spin_unlock(&s->bucket_lock);

	time = jiffies_to_msecs(jiffies - time);

	stats.count++;
	stats.ms_max	= max_t(unsigned, stats.ms_max, time);
	stats.last	= get_seconds();

	stats.key_bytes *= sizeof(uint64_t);
	stats.dirty	<<= 9;
	stats.data	<<= 9;
	memcpy(&s->gc_stats, &stats, sizeof(struct gc_stat));

	pr_debug("gc took %lu ms, %li pinned, %i%% used, %zu btree "
		 "nodes %i%% used, need_gc was %i now %i", time, pinned,
		 in_use(s), stats.nodes, btree_used(s), need_gc, s->need_gc);
out:
	closure_sync(&sr.insert);
}

static void btree_gc_work(struct work_struct *w)
{
	struct cache_set *c = container_of(w, struct cache_set, gc_work);

	if (!mutex_trylock(&c->gc_lock))
		return;

	btree_gc(c);
	mutex_unlock(&c->gc_lock);
}

/* Btree insertion */

static void shift_keys(struct bset *i, struct bkey *where, struct bkey *insert)
{
	size_t len = (void *) end(i) - (void *) where;
	int n = 2 + KEY_PTRS(insert);
	i->keys += n;
	BUG_ON(len && !KEY_IS_HEADER(where));

	memmove((uint64_t *) where + n, where, len);
	bkey_copy(where, insert);
}

static bool check_old_keys(struct btree *b, struct bkey *k,
			   struct btree_iter *iter, struct search *s)
{
	bool overwrote = false;

	while (1) {
		struct bkey *j = btree_iter_next(iter);
		if (!j || bkey_cmp(k, &START_KEY(j)) <= 0)
			break;

		if (s->insert_type == INSERT_READ &&
		    !ptr_bad(b, j)) {
			/* Should split this key in two if necessary */
			if (bkey_cmp(&START_KEY(j), &START_KEY(k)) > 0)
				cut_back(&START_KEY(j), k);
			else if (bkey_cmp(j, k) < 0)
				cut_front(j, k);
			else {
				atomic_long_inc(&s->d->cache_miss_collisions);
				return true;
			}

			BUG_ON(!KEY_SIZE(k));
			continue;
		}

		if (s->insert_type == INSERT_UNDIRTY) {
			if (j->header != (k->header | PTR_DIRTY_BIT) ||
			    memcmp(&j->key, &k->key, key_bytes(k) - 8))
				goto wb_failed;

			cut_back(&START_KEY(k), j);
			atomic_long_inc(&b->c->writeback_keys_done);
			return false;
		}

		if (bkey_cmp(k, j) < 0) {
			if (bkey_cmp(&START_KEY(k), &START_KEY(j)) > 0) {
				struct bset *w = write_block(b);
				struct bkey *m = j;

				if (j < w->start) {
					m = btree_bsearch(w, k);
					shift_keys(w, m, j);
				} else {
					shift_keys(w, m, j);
					m = next(j);
				}

				cut_front(k, m);
				cut_back(&START_KEY(k), j);
				return false;
			}

			cut_front(k, j);
		} else
			__cut_back(&START_KEY(k), j);

		overwrote = true;
	}

	if (s->insert_type == INSERT_UNDIRTY) {
wb_failed:	atomic_long_inc(&b->c->writeback_keys_failed);
		return true;
	}

	if (!KEY_PTRS(k) && !overwrote)
		return true;

	return false;
}

static bool btree_insert_keys(struct btree *b, struct search *s)
{
	bool merge(struct bset *i, struct bkey *m, struct bkey *k)
	{
		if (m != i->start) {
			m = prev(m);

			if (KEY_PTRS(m) == KEY_PTRS(k) && !KEY_SIZE(m))
				goto copy;

			if (btree_try_merge(b, m, k))
				return true;

			m = next(m);
		}

		if (m != end(i)) {
			if (KEY_PTRS(m) == KEY_PTRS(k) && !KEY_SIZE(m))
				goto copy;

			if (btree_try_merge(b, k, m))
				return true;
		}

		return false;
copy:
		bkey_copy(m, k);
		return true;
	}

	/* If a read generates a cache miss, and a write to the same location
	 * finishes before the new data is added to the cache, the write will
	 * be overwritten with stale data. We can catch this by never
	 * overwriting good data if it came from a read.
	 */
	bool ret = false;
	struct bset *i = write_block(b);
	struct bkey *k, *m;

	while ((k = keylist_pop(&s->keys))) {
		const char *status = "replacing";
		int oldsize = count_data(b);

		BUG_ON(b->level && !KEY_PTRS(k));
		BUG_ON(!b->level && !k->key);
		BUG_ON(!b->level && s->insert_type != INSERT_REPLAY &&
		       (!KEY_DIRTY(k) == (s->insert_type == INSERT_WRITEBACK)));

		bkey_put(b->c, k, s->insert_type, b->level);

		if (!b->level) {
			struct btree_iter iter;
			m = btree_iter_init(b, &iter, &START_KEY(k), 0);

			BUG_ON(!m || m < i->start);

			if (check_old_keys(b, k, &iter, s))
				continue;

			while (m != end(i) && bkey_cmp(k, &START_KEY(m)) > 0)
				m = next(m);

			if (merge(i, m, k))
				goto merged;
		} else
			m = btree_bsearch(i, k);

		status = "inserting";
		shift_keys(i, m, k);
merged:
		check_key_order_msg(b, i, "was last %s %s", status, pkey(k));
		BUG_ON(count_data(b) < oldsize);
		ret = true;

		if (b->level && !k->key)
			b->prio_blocked++;

		pr_debug("%s for %s at %s block %i key %zu/%i: %s",
			 status, insert_type(s), pbtree(b),
			 index(i, b), m - i->start, i->keys, pkey(k));
	}

	return ret;
}

static int btree_split(struct btree *b, struct search *s)
{
	bool split, root = b == b->c->root;
	struct btree *n1, *n2 = NULL, *n3 = NULL;

	n1 = btree_alloc(b->c, b->level, &s->insert);
	if (IS_ERR(n1))
		goto err;

	btree_sort(b, 0, n1->data);
	bkey_copy_key(&n1->key, &b->key);

	split = set_blocks(n1->data, n1->c) > (btree_blocks(b) * 3) / 4;
	pr_debug("%ssplitting at %s keys %i", split ? "" : "not ",
		 pbtree(b), n1->data->keys);

	if (split) {
		n2 = btree_alloc(b->c, b->level, &s->insert);
		if (IS_ERR(n2))
			goto err_free1;

		if (root) {
			n3 = btree_alloc(b->c, b->level + 1, &s->insert);
			if (IS_ERR(n3))
				goto err_free2;
		}

		btree_insert_keys(n1, s);

		n2->data->keys = (n1->data->keys * 2) / 5;

		while (!KEY_IS_HEADER(node(n1->data, n1->data->keys
					   - n2->data->keys)))
			n2->data->keys++;

		n1->data->keys -= n2->data->keys;

		memcpy(n2->data->start,
		       end(n1->data),
		       n2->data->keys * sizeof(uint64_t));

		bkey_copy_key(&n1->key, last_key(n1->data));
		bkey_copy_key(&n2->key, &b->key);

		keylist_add(&s->keys, &n2->key);
		btree_write(n2, true, s);
		rw_unlock(true, n2);
	} else
		btree_insert_keys(n1, s);

	keylist_add(&s->keys, &n1->key);
	btree_write(n1, true, s);

	if (n3) {
		bkey_copy_key(&n3->key, &MAX_KEY);
		btree_insert_keys(n3, s);
		btree_write(n3, true, s);

		closure_sync(&s->insert);
		set_new_root(n3);
		rw_unlock(true, n3);
	} else if (root) {
		s->keys.top = s->keys.bottom;
		closure_sync(&s->insert);
		set_new_root(n1);
	} else {
		bkey_copy(s->keys.top, &b->key);
		bkey_copy_key(s->keys.top, &ZERO_KEY);

		for (unsigned i = 0; i < KEY_PTRS(&b->key); i++) {
			uint8_t g = PTR_BUCKET(b->c, &b->key, i)->gen + 1;

			s->keys.top->ptr[i] -= PTR_GEN(s->keys.top, i);
			s->keys.top->ptr[i] += g;
		}

		keylist_push(&s->keys);
		closure_sync(&s->insert);
		atomic_inc(&b->c->prio_blocked);
	}

	rw_unlock(true, n1);
	btree_free(b, s);

	return 0;
err_free2:
	__bkey_put(n2->c, &n2->key);
	btree_free(n2, s);
	rw_unlock(true, n2);
err_free1:
	__bkey_put(n1->c, &n1->key);
	btree_free(n1, s);
	rw_unlock(true, n1);
err:
	if (n3 == ERR_PTR(-EAGAIN) ||
	    n2 == ERR_PTR(-EAGAIN) ||
	    n1 == ERR_PTR(-EAGAIN))
		return -EAGAIN;

	printk(KERN_WARNING "bcache: couldn't split");
	return -ENOMEM;
}

static int btree_insert(struct btree *b, struct search *s)
{
	if (should_split(b)) {
		if (s->lock <= b->c->root->level) {
			BUG_ON(b->level);
			s->lock = b->c->root->level + 1;
			return -EINTR;
		}
		return btree_split(b, s);
	}

	if (write_block(b) != b->sets[b->nsets]) {
		struct bset *i;
		unsigned keys = 0;

		for_each_sorted_set(b, i)
			keys += i->keys;

		if (b->nsets > 3)
			for (unsigned j = 0; b->nsets - j > 2; j++) {
				if (keys > b->sets[j]->keys * 2 ||
				    keys < 100) {
					btree_sort(b, j, NULL);
					break;
				}

				keys -= b->sets[j]->keys;
			}

		if (b->nsets > max(2, 4 - b->level))
			btree_sort(b, 0, NULL);

		i = write_block(b);
		bset_init(b, i);
		b->sets[++b->nsets] = i;
	}

	if (btree_insert_keys(b, s))
		btree_write(b, false, s);

	return 0;
}

static int btree_insert_recurse(struct btree *b, struct search *s,
				struct keylist *stack_keys)
{
	if (b->level) {
		int ret;
		struct bkey *insert = s->keys.bottom;
		struct bkey *k = next_recurse_key(b, &START_KEY(insert));

		if (!k) {
			cache_bug(b, "no key to recurse on at level %i/%i",
				  b->level, b->c->root->level);

			s->keys.top = s->keys.bottom;
			return -EIO;
		}

		if (bkey_cmp(insert, k) > 0) {
			if (s->insert_type == INSERT_UNDIRTY) {
				s->keys.top = s->keys.bottom;
				return 0;
			}

			for (unsigned i = 0; i < KEY_PTRS(insert); i++)
				atomic_inc(&PTR_BUCKET(b->c, insert, i)->pin);

			bkey_copy(stack_keys->top, insert);

			cut_back(k, insert);
			cut_front(k, stack_keys->top);

			keylist_push(stack_keys);
		}

		ret = btree(insert_recurse, b, k, s, stack_keys);
		if (ret)
			return ret;
	}

	if (!keylist_empty(&s->keys)) {
		return btree_insert(b, s);
	}

	return 0;
}

static void __btree_insert_async(struct search *s, struct cache_set *c)
{
	int ret;
	struct cache *ca;
	struct keylist stack_keys;

	if (s->insert_type == INSERT_READ)
		__bio_complete(s);

	set_bit(CLOSURE_BLOCK, &s->insert.flags);

	BUG_ON(keylist_empty(&s->keys));
	keylist_copy(&stack_keys, &s->keys);
	keylist_init(&s->keys);

	while (c->need_gc > MAX_NEED_GC) {
		mutex_lock(&c->gc_lock);

		if (c->need_gc > MAX_NEED_GC)
			btree_gc(c);

		mutex_unlock(&c->gc_lock);
	}

	for_each_cache(ca, c)
		while (ca->need_save_prio > MAX_SAVE_PRIO) {
			spin_lock(&c->bucket_lock);
			free_some_buckets(ca);
			spin_unlock(&c->bucket_lock);

			closure_wait_on(&c->bucket_wait, delayed, &s->insert,
					ca->need_save_prio <= MAX_SAVE_PRIO ||
					(atomic_read(&ca->prio_written) >= 0 &&
					 atomic_read(&c->prio_blocked) == 0));
		}

	while (!keylist_empty(&stack_keys) ||
	       !keylist_empty(&s->keys)) {
		if (keylist_empty(&s->keys)) {
			keylist_add(&s->keys, keylist_pop(&stack_keys));
			s->lock = 0;
		}

		ret = btree_root(insert_recurse, c, s, &stack_keys);

		if (ret == -EAGAIN)
			closure_sync(&s->insert);
		else if (ret) {
			struct bkey *k;

			printk(KERN_WARNING "bcache: error %i trying to "
			       "insert key for %s\n", ret, insert_type(s));

			s->error = -ENOMEM;

			while ((k = keylist_pop(&stack_keys) ?:
				    keylist_pop(&s->keys)))
				bkey_put(c, k, s->insert_type, 0);
		}
	}

	keylist_free(&stack_keys);

	if (s->journal)
		atomic_dec_bug(s->journal);
	s->journal = NULL;
}

static void btree_insert_async(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, insert);
again:	__btree_insert_async(s, s->d->c);

	if (s->skip && !s->bio_done) {
		btree_invalidate(s);
		goto again;
	}

	return_f(cl, !s->bio_done
		 ? bio_insert : NULL);
}

static void btree_invalidate(struct search *s)
{
	struct bio *bio = s->cache_bio;

	pr_debug("invalidating %i sectors from %llu",
		 bio_sectors(bio), (uint64_t) bio->bi_sector);

	s->insert.fn = btree_journal;

	while (bio_sectors(bio)) {
		unsigned len = min(bio_sectors(bio), 1U << 14);
		if (keylist_realloc(&s->keys, 0))
			return;

		bio->bi_sector	+= len;
		bio->bi_size	-= len << 9;

		keylist_add(&s->keys, &KEY(s->d->id, bio->bi_sector, len));
	}

	s->bio_done = true;
}

/* Journalling */

static void btree_journal_read_endio(struct bio *bio, int error)
{
	struct closure *c = bio->bi_private;
	bio_put(bio);
	closure_put(c, delayed);
}

static int btree_journal_read(struct cache_set *s, struct list_head *list,
			      struct search *sr)
{
	struct cache *c;
	struct closure *cl = &sr->insert;
	struct journal_replay *i;
	struct jset *j, *data = s->journal.w[0].data;

	for_each_cache(c, s) {
		struct bio *bio = &c->journal_bio;
		sector_t offset = c->journal_area_start;
		unsigned len, left;

		while (offset < c->journal_area_end) {
reread:			left = c->journal_area_end - offset;
			len = min_t(unsigned, left, PAGE_SECTORS * 8);

			bio_reset(bio);
			bio->bi_sector	= offset;
			bio->bi_bdev	= c->bdev;
			bio->bi_rw	= READ;
			bio->bi_size	= len << 9;

			bio->bi_end_io	= btree_journal_read_endio;
			bio->bi_private = cl;
			bio_map(bio, data);

			closure_get(cl);
			closure_bio_submit(bio, cl, s->bio_split);
			closure_sync(cl);

			j = data;
			while (len) {
				struct list_head *where = list;
				size_t blocks = 1, bytes = set_bytes(j);

				if (j->magic != jset_magic(s))
					goto next_set;

				if (bytes > left << 9)
					goto next_set;

				if (bytes > len << 9)
					goto reread;

				if (j->csum != csum_set(j))
					goto next_set;

				blocks = set_blocks(j, s);

				while (!list_empty(list)) {
					i = list_first_entry(list,
						struct journal_replay, list);
					if (i->j.seq >= j->last_seq)
						break;
					list_del(&i->list);
					kfree(i);
				}

				list_for_each_entry_reverse(i, list, list) {
					if (j->seq == i->j.seq)
						goto next_set;

					if (j->seq < i->j.last_seq)
						goto next_set;

					if (j->seq > i->j.seq) {
						where = &i->list;
						break;
					}
				}

				i = kmalloc(offsetof(struct journal_replay, j) +
					    bytes, GFP_KERNEL);
				if (!i)
					return -ENOMEM;
				memcpy(&i->j, j, bytes);
				list_add(&i->list, where);
next_set:
				offset	+= blocks * s->sb.block_size;
				len	-= blocks * s->sb.block_size;
				j = ((void *) j) + blocks * block_bytes(s);
			}
		}
	}

	return 0;
}

static void btree_journal_mark(struct cache_set *c, struct list_head *list)
{
	struct journal_replay *i;

	list_for_each_entry(i, list, list)
		for (struct bkey *k = i->j.start; k < end(&i->j); k = next(k)) {
			for (unsigned j = 0; j < KEY_PTRS(k); j++) {
				struct bucket *g = PTR_BUCKET(c, k, j);
				atomic_inc(&g->pin);

				if (g->prio == btree_prio)
					g->prio = initial_prio;
				BUG_ON(ptr_stale(c, k, j));
			}

			__btree_mark_key(c, 0, k);
		}
}

static int btree_journal_replay(struct cache_set *s, struct list_head *list,
				struct search *sr)
{
	int ret = 0, keys = 0, entries = 0;
	atomic_t p = { 0 };
	struct journal_replay *i =
		list_entry(list->prev, struct journal_replay, list);

	uint64_t start = i->j.last_seq, end = i->j.seq, last = start - 1;

	sr->insert_type = INSERT_REPLAY;

	list_for_each_entry(i, list, list) {
		BUG_ON(last + 1 > i->j.seq);
		if (last + 1 != i->j.seq)
			printk(KERN_ERR "bcache: journal entries %llu-%llu "
			       "missing! (replaying %llu-%llu)\n",
			       last, i->j.seq, start, end);

		/* XXX: Need to refcount journal entries.. or flush btree after
		 * replay is done?
		 */
		BUG_ON(!fifo_push(&s->journal.pin, p));
		atomic_set(&fifo_back(&s->journal.pin), 0);

		for (struct bkey *k = i->j.start; k < end(&i->j); k = next(k)) {
			pr_debug("%s", pkey(k));
			bkey_copy(sr->keys.top, k);
			keylist_push(&sr->keys);

			sr->journal = &fifo_back(&s->journal.pin);
			atomic_inc(sr->journal);

			__btree_insert_async(sr, s);
			BUG_ON(!keylist_empty(&sr->keys));
			keys++;
		}
		last = i->j.seq;
		entries++;
	}

	printk(KERN_NOTICE "bcache: journal replay done, %i keys in %i "
	       "entries, seq %llu-%llu\n", keys, entries, start, end);

	while (!list_empty(list)) {
		i = list_first_entry(list, struct journal_replay, list);
		list_del(&i->list);
		kfree(i);
	}

	closure_sync(&sr->insert);
	return ret;
}

void btree_flush_write(struct cache_set *s)
{
	struct btree *b, *i;

	while (!s->root->level) {
		i = s->root;
		rw_lock(true, i, 0);

		if (i == s->root)
			goto found;
		rw_unlock(true, i);
	}

	spin_lock(&s->bucket_lock);

	i = NULL;
	list_for_each_entry(b, &s->lru, lru) {
		if (!down_write_trylock(&b->lock))
			continue;

		if (!b->write || !b->write->journal)
			goto next;

		if (i && journal_pin_cmp(s, i->write, b->write)) {
			rw_unlock_nowrite(true, i);
			i = NULL;
		}

		if (!i) {
			i = b;
			continue;
		}
next:
		rw_unlock_nowrite(true, b);
	}

	if (!i) {
		/* We can't find the best btree, just pick the first */
		list_for_each_entry(b, &s->lru, lru)
			if (!b->level && b->write) {
				i = b;
				break;
			}

		spin_unlock(&s->bucket_lock);
		if (!i)
			return;

		rw_lock(true, i, i->level);
	} else
		spin_unlock(&s->bucket_lock);
found:
	i->expires = jiffies;
	if (i->work.timer.function)
		mod_timer_pending(&i->work.timer, i->expires);

	rw_unlock(true, i);
	pr_debug("");
}

static void btree_journal_alloc(struct cache_set *s)
{
	struct journal_write *w = s->journal.cur;
	struct cache *c;
	unsigned n = 0, free;

	s->journal.sectors_free = UINT_MAX;

	/* XXX: Sort by free journal space */

	for_each_cache(c, s) {
		if (c->journal_start == c->journal_end)
			continue;

		if (c->journal_start == c->journal_area_end)
			c->journal_start = c->journal_area_start;

		w->key.ptr[n++] = PTR(0, c->journal_start, c->sb.nr_this_dev);

		free = c->journal_start < c->journal_end
			? c->journal_end
			: c->journal_area_end;
		free -= c->journal_start;
		BUG_ON(!free);

		s->journal.sectors_free = min(s->journal.sectors_free, free);
	}

	if (!n)
		printk(KERN_NOTICE "bcache: journal filled up\n");
	else
		closure_run_wait(&w->c->journal.wait, delayed);
	w->key.header = KEY_HEADER(0, 0);
	SET_KEY_PTRS(&w->key, n);
}

static bool journal_full(struct cache_set *c)
{
	return !KEY_PTRS(&c->journal.cur->key) || fifo_full(&c->journal.pin);
}

#define last_seq(j)	((j)->seq - fifo_used(&(j)->pin) + 1)

static void btree_journal_reclaim(struct cache_set *s)
{
	struct cache *c;
	struct journal_seq j;
	atomic_t p;
	bool popped = false, full = journal_full(s);

	while (fifo_used(&s->journal.pin) > 1 &&
	       !atomic_read(&fifo_front(&s->journal.pin))) {
		fifo_pop(&s->journal.pin, p);
		popped = true;
	}

	if (!popped)
		return;

	if (full)
		pr_debug("journal_pin popped");

	for_each_cache(c, s)
		while (!fifo_empty(&c->journal) &&
		       fifo_front(&c->journal).sector != c->journal_start &&
		       fifo_front(&c->journal).seq < last_seq(&s->journal)) {
			fifo_pop(&c->journal, j);
			c->journal_end = j.sector;
		}

	if (!KEY_PTRS(&s->journal.cur->key)) {
		pr_debug("allocating");
		btree_journal_alloc(s);
	}

	closure_run_wait(&s->journal.wait, delayed);
}

static void __btree_journal_meta(struct cache_set *c)
{
	struct cache *ca;
	struct journal_write *w = c->journal.cur;

	w->data->btree_level = c->root->level;

	bkey_copy(&w->data->btree_root, &c->root->key);
	bkey_copy(&w->data->uuid_bucket, &c->uuid_bucket);

	for_each_cache(ca, c)
		w->data->prio_bucket[ca->sb.nr_this_dev] = ca->prio_start;

	w->data->magic = jset_magic(c);
}

static void btree_journal_next(struct cache_set *s)
{
	struct journal_write *w = s->journal.cur;
	atomic_t p = { 0 };

	for (unsigned i = 0; i < KEY_PTRS(&w->key); i++) {
		struct cache *c = PTR_CACHE(s, &w->key, i);
		struct journal_seq seq = { .seq = w->data->seq };

		c->journal_start += set_blocks(w->data, s) * s->sb.block_size;
		BUG_ON(c->journal_start > c->journal_area_end);

		seq.sector = c->journal_start;
		BUG_ON(!fifo_push(&c->journal, seq));
	}

	w = s->journal.cur = w == s->journal.w
		? &s->journal.w[1]
		: &s->journal.w[0];

	BUG_ON(!fifo_push(&s->journal.pin, p));
	atomic_set(&fifo_back(&s->journal.pin), 0);

	w->need_write		= false;
	w->data->keys		= 0;
	w->data->seq		= ++s->journal.seq;
	w->data->last_seq	= last_seq(&s->journal);

	__btree_journal_meta(s);

	if (fifo_full(&s->journal.pin))
		pr_debug("journal_pin full (%zu)", fifo_used(&s->journal.pin));

	btree_journal_alloc(s);
}

static void btree_journal_endio(struct bio *bio, int error)
{
	struct journal_write *w = bio->bi_private;
	bio_put(bio);

	cache_set_err_on(error, w->c, "journal io error");

	if (!atomic_dec_and_test(&w->c->journal.io))
		return;

	closure_run_wait(&w->wait, delayed);
	atomic_set(&w->c->journal.io, -1);
	schedule_work(&w->c->journal.work);
}

static void btree_journal_write(struct cache_set *s, struct journal_write *w)
{
	w->data->csum = csum_set(w->data);

	for (unsigned i = 0; i < KEY_PTRS(&w->key); i++) {
		struct cache *c = PTR_CACHE(s, &w->key, i);
		struct bio *bio = &c->journal_bio;

		atomic_long_add(set_blocks(w->data, s) * c->sb.block_size,
				&c->meta_sectors_written);

		bio_reset(bio);
		bio->bi_sector	= PTR_OFFSET(&w->key, i);
		bio->bi_bdev	= c->bdev;
		bio->bi_rw	= REQ_META|WRITE_SYNC;
		bio->bi_size	= set_blocks(w->data, s) * block_bytes(s);

		bio->bi_end_io	= btree_journal_endio;
		bio->bi_private = w;
		bio_map(bio, w->data);

		pr_debug("write to sector %lu", bio->bi_sector);
		atomic_inc(&s->journal.io);
		bio_submit_split(bio, &s->journal.io, s->bio_split);
	}
}

static void __btree_journal_try_write(struct cache_set *c, bool noflush)
{
	struct journal_write *w = c->journal.cur;

	if (!w->need_write)
		spin_unlock(&c->journal.lock);
	else if (journal_full(c)) {
		btree_journal_reclaim(c);
		spin_unlock(&c->journal.lock);

		if (!noflush)
			btree_flush_write(c);
		schedule_work(&c->journal.work);
	} else if (atomic_cmpxchg(&c->journal.io, -1, 0) == -1) {
		btree_journal_next(c);
		spin_unlock(&c->journal.lock);
		btree_journal_write(c, w);
	} else
		spin_unlock(&c->journal.lock);
}

#define btree_journal_try_write(c)	__btree_journal_try_write(c, false)

static void btree_journal_work(struct work_struct *work)
{
	struct journal *j = container_of(work, struct journal, work);
	struct cache_set *c = container_of(j, struct cache_set, journal);

	spin_lock(&c->journal.lock);
	btree_journal_try_write(c);
}

static void btree_journal_wait(struct cache_set *c, struct closure *cl)
{
	struct journal_write *w;

	spin_lock(&c->journal.lock);
	w = c->journal.cur;
	if (w->need_write)
		BUG_ON(!closure_wait(&w->wait, cl));

	btree_journal_try_write(c);
}

static void btree_journal_meta(struct cache_set *c, struct closure *cl)
{
	if (CACHE_SYNC(&c->sb)) {
		spin_lock(&c->journal.lock);
		c->journal.cur->need_write = true;

		if (cl)
			BUG_ON(!closure_wait(&c->journal.cur->wait, cl));

		__btree_journal_meta(c);
		__btree_journal_try_write(c, true);
	}
}

static void btree_journal(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, insert);
	struct cache_set *c = s->d->c;
	struct journal_write *w;
	size_t b, n = ((uint64_t *) s->keys.top) - s->keys.list;

	if (!(s->insert_type & INSERT_WRITE) ||
	    !CACHE_SYNC(&s->d->c->sb))
		goto out;

	spin_lock(&c->journal.lock);

	btree_journal_reclaim(c);

	if (journal_full(c)) {
		/* XXX: tracepoint */
		BUG_ON(!closure_wait(&c->journal.wait, cl));
		spin_unlock(&c->journal.lock);

		btree_flush_write(c);
		return_f(cl, btree_journal);
	}

	w = c->journal.cur;
	b = __set_blocks(w->data, w->data->keys + n, c);

	if (b * c->sb.block_size > PAGE_SECTORS << JSET_BITS ||
	    b * c->sb.block_size > c->journal.sectors_free) {
		/* XXX: tracepoint */
		BUG_ON(!closure_wait(&w->wait, cl));

		btree_journal_try_write(c);
		return_f(cl, btree_journal);
	}

	memcpy(end(w->data), s->keys.list, n * sizeof(uint64_t));
	w->data->keys += n;
	w->need_write = true;

	s->journal = &fifo_back(&c->journal.pin);
	atomic_inc(s->journal);

	/* XXX: if bio_insert doesn't finish on the first loop through this may
	 * bug
	 */
	BUG_ON(!closure_wait(&w->wait, &s->cl));

	btree_journal_try_write(c);
out:
	btree_insert_async(cl);
}

static void set_new_root(struct btree *b)
{
	BUG_ON(!b->written);

	for (unsigned i = 0; i < KEY_PTRS(&b->key); i++)
		BUG_ON(PTR_BUCKET(b->c, &b->key, i)->prio != btree_prio);

	spin_lock(&b->c->bucket_lock);
	list_del_init(&b->lru);
	spin_unlock(&b->c->bucket_lock);

	b->c->root = b;
	__bkey_put(b->c, &b->key);

	btree_journal_meta(b->c, NULL);
	pr_debug("%s for %pf", pbtree(b), __builtin_return_address(0));
}

/* UUID io */

static void uuid_endio(struct bio *bio, int error)
{
	struct cache *c = bio->bi_private;
	cache_err_on(error, c, "accessing uuids");

	bio_put(bio);
	closure_put(c->set->uuid_writer, delayed);
}

static void uuid_io(struct cache_set *c, unsigned long rw,
		    struct bkey *k, struct closure *cl)
{
	lockdep_assert_held(&register_lock);
	/* XXX: check for io errors */
	c->uuid_writer = cl;

	for (unsigned i = 0; i < KEY_PTRS(k); i++) {
		struct cache *ca = PTR_CACHE(c, k, i);
		struct bio *bio = ca->uuid_bio;

		bio_reset(bio);
		bio->bi_sector	= PTR_OFFSET(k, i);
		bio->bi_bdev	= ca->bdev;
		bio->bi_rw	= REQ_META|rw;
		bio->bi_size	= KEY_SIZE(k) << 9;

		bio->bi_end_io	= uuid_endio;
		bio->bi_private = ca;
		bio_map(bio, c->uuids);

		closure_get(cl);
		closure_bio_submit(bio, cl, c->bio_split);

		if (!(rw & WRITE))
			break;
	}

	pr_debug("%s UUIDs at %s", rw & WRITE ? "wrote" : "read",
		 pkey(&c->uuid_bucket));
	for (struct uuid_entry *u = c->uuids; u < c->uuids + c->nr_uuids; u++)
		if (!is_zero(u->uuid, 16))
			pr_debug("Slot %zi: %pU: %s: 1st: %u last: %u inv: %u",
				 u - c->uuids, u->uuid, u->label,
				 u->first_reg, u->last_reg, u->invalidated);
}

static int uuid_write(struct cache_set *c)
{
	BKEY_PADDED(key) k;
	struct closure cl;
	closure_init_stack(&cl);

	lockdep_assert_held(&register_lock);

	if (pop_bucket_set(c, btree_prio, &k.key, 1, &cl))
		return 1;

	SET_KEY_SIZE(&k.key, c->sb.bucket_size);
	uuid_io(c, WRITE_SYNC, &k.key, &cl);
	closure_sync(&cl);

	bkey_copy(&c->uuid_bucket, &k.key);
	__bkey_put(c, &k.key);

	btree_journal_meta(c, NULL);
	return 0;
}

/* Background writeback */

static void dirty_init(struct dirty *w)
{
	struct bio *bio = &w->io->bio;

	bio_init(bio);
	bio_get(bio);
	bio_set_prio(bio, IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0));

	bio->bi_size		= KEY_SIZE(&w->key) << 9;
	bio->bi_private		= w;
	bio_map(bio, NULL);
}

static int dirty_cmp(struct dirty *r, struct dirty *l)
{
	/* Overlapping keys must compare equal */
	if (KEY_START(&r->key) >= l->key.key)
		return 1;
	if (KEY_START(&l->key) >= r->key.key)
		return -1;
	return 0;
}

#define WRITEBACK_SLURP	100

static int btree_refill_dirty(struct btree *b, struct search *s, int *count)
{
	struct dirty *w;
	struct btree_iter iter;
	btree_iter_init(b, &iter, &KEY(s->d->id, s->d->last_found, 0), 0);

	/* To protect rb tree access vs. read_dirty() */
	if (!b->level)
		spin_lock(&s->d->lock);

	while (*count < WRITEBACK_SLURP) {
		struct bkey *k = btree_iter_next(&iter);
		if (!k || KEY_DEV(k) != s->d->id)
			break;

		if (ptr_bad(b, k))
			continue;

		if (b->level) {
			int ret = btree(refill_dirty, b, k, s, count);
			if (ret)
				return ret;

		} else if (KEY_DIRTY(k)) {
			w = kmem_cache_alloc(dirty_cache, GFP_NOWAIT);
			if (!w) {
				spin_unlock(&s->d->lock);

				w = kmem_cache_alloc(dirty_cache, GFP_NOIO);
				if (!w)
					return -ENOMEM;

				spin_lock(&s->d->lock);
			}

			s->d->last_found = k->key;
			pr_debug("%s", pkey(k));
			w->io = NULL;
			bkey_copy(&w->key, k);
			SET_KEY_DIRTY(&w->key, false);

			if (RB_INSERT(&s->d->dirty, w, node, dirty_cmp))
				kmem_cache_free(dirty_cache, w);
			else
				(*count)++;
		}
	}

	if (!b->level)
		spin_unlock(&s->d->lock);

	return 0;
}

static bool refill_dirty(struct cached_dev *d)
{
	int r, count = 0;
	struct search s;
	uint64_t l;

	search_init_stack(&s);
	s.d = d;

	down_write(&d->writeback_lock);
again:
	l = d->last_found;
	r = btree_root(refill_dirty, d->c, &s, &count);

	pr_debug("found %i keys on %i from %llu to %llu, %i%% used",
		 count, d->id, l, d->last_found, in_use(d->c));

	if (!r && count < WRITEBACK_SLURP) {
		/* Got to the end of the btree */
		d->last_found = 0;

		if (l)
			goto again;

		/* Scanned the whole thing */
		if (!count && !atomic_read(&d->in_flight)) {
			if (!d->writeback &&
			    BDEV_STATE(&d->sb) == BDEV_STATE_DIRTY) {
				SET_BDEV_STATE(&d->sb, BDEV_STATE_CLEAN);
				write_bdev_super(d, NULL);
			}

			atomic_long_set(&d->last_refilled, 0);
		} else
			atomic_long_set(&d->last_refilled, jiffies);
	}

	up_write(&d->writeback_lock);
	closure_sync(&s.insert);

	return count;
}

static bool in_writeback(struct cached_dev *d, sector_t offset, unsigned len)
{
	struct dirty *ret, s;
	s.key = KEY(d->id, offset + len, len);

	spin_lock(&d->lock);
	ret = RB_SEARCH(&d->dirty, s, node, dirty_cmp);
	if (ret && !ret->io) {
		rb_erase(&ret->node, &d->dirty);
		kmem_cache_free(dirty_cache, ret);
		ret = NULL;
	}

	spin_unlock(&d->lock);
	return ret;
}

static void queue_writeback(struct cached_dev *d)
{
	atomic_inc(&d->count);
	if (!queue_work(writeback, &d->refill))
		cached_dev_put(d);
}

static bool should_refill_dirty(struct cached_dev *d)
{
	long t = atomic_long_read(&d->last_refilled);
	unsigned ms = d->writeback_delay * 1000;

	return t &&
		((d->writeback_running &&
		  ((jiffies_to_msecs(jiffies - t) > ms &&
		    in_use(d->c) > d->writeback_percent) ||
		   !d->writeback)) ||
		 atomic_read(&d->closing));
}

static void maybe_refill_dirty(struct search *s)
{
	if (s->insert_type != INSERT_READ &&
	    !atomic_read(&s->d->in_flight)) {
		if (should_refill_dirty(s->d))
			queue_writeback(s->d);

		if (s->insert_type == INSERT_WRITEBACK)
			atomic_long_cmpxchg(&s->d->last_refilled, 0, jiffies);
	}
}

static void write_dirty_finish(struct closure *cl)
{
	struct dirty_io *io = container_of(cl, struct dirty_io, cl);
	struct dirty *w = io->bio.bi_private;
	struct cached_dev *d = io->d;
	struct bio_vec *bv = bio_iovec_idx(&io->bio, io->bio.bi_vcnt);

	while (bv-- != w->io->bio.bi_io_vec)
		__free_page(bv->bv_page);

	closure_del(cl);
	kfree(io);

	if (!KEY_DIRTY(&w->key)) {
		struct search s;
		search_init_stack(&s);

		s.insert_type = INSERT_UNDIRTY;
		keylist_add(&s.keys, &w->key);

		pr_debug("clearing %s", pkey(&w->key));
		__btree_insert_async(&s, d->c);
		closure_sync(&s.insert);
	}

	spin_lock(&d->lock);
	rb_erase(&w->node, &d->dirty);
	kmem_cache_free(dirty_cache, w);
	atomic_dec_bug(&d->in_flight);

	read_dirty(d);
}

static void dirty_endio(struct bio *bio, int error)
{
	struct dirty *w = bio->bi_private;

	if (error)
		SET_KEY_DIRTY(&w->key, true);

	bio_put(bio);
	closure_put(&w->io->cl, writeback);
}

static void write_dirty(struct closure *cl)
{
	struct dirty_io *io = container_of(cl, struct dirty_io, cl);
	struct dirty *w = io->bio.bi_private;

	dirty_init(w);
	io->cl.fn		= write_dirty_finish;
	io->bio.bi_rw		= WRITE|REQ_UNPLUG;
	io->bio.bi_sector	= KEY_START(&w->key);
	io->bio.bi_bdev		= io->d->bdev;
	io->bio.bi_end_io	= dirty_endio;

	closure_bio_submit(&w->io->bio, cl, io->d->c->bio_split);
}

static void read_dirty_endio(struct bio *bio, int error)
{
	struct dirty *w = bio->bi_private;
	cache_err_on(error, PTR_CACHE(w->io->d->c, &w->key, 0),
		      "reading from cache");

	dirty_endio(bio, error);
}

static void read_dirty(struct cached_dev *d)
{
	while (d->writeback_running) {
		struct dirty *w, s;
		s.key = KEY(d->id, d->last_read, 0);

		w = RB_GREATER(&d->dirty, s, node, dirty_cmp) ?:
		    RB_FIRST(&d->dirty, struct dirty, node);

		if (!w || w->io) {
			spin_unlock(&d->lock);

			if (should_refill_dirty(d) &&
			    refill_dirty(d)) {
				spin_lock(&d->lock);
				continue;
			}

			goto out;
		}

		if (ptr_stale(d->c, &w->key, 0)) {
			rb_erase(&w->node, &d->dirty);
			kmem_cache_free(dirty_cache, w);
			continue;
		}

		w->io = ERR_PTR(-EINTR);
		spin_unlock(&d->lock);

		w->io = kzalloc(sizeof(struct dirty_io) + sizeof(struct bio_vec)
				* DIV_ROUND_UP(KEY_SIZE(&w->key), PAGE_SECTORS),
				GFP_KERNEL);
		if (!w->io)
			goto out;

		closure_init(&w->io->cl, NULL);
		w->io->cl.fn		= write_dirty;
		w->io->d		= d;

		dirty_init(w);
		w->io->bio.bi_sector	= PTR_OFFSET(&w->key, 0);
		w->io->bio.bi_bdev	= PTR_CACHE(d->c, &w->key, 0)->bdev;
		w->io->bio.bi_rw	= READ|REQ_UNPLUG;
		w->io->bio.bi_end_io	= read_dirty_endio;

		if (bio_alloc_pages(&w->io->bio, GFP_KERNEL)) {
			kfree(w->io);
			w->io = NULL;
			goto out;
		}

		d->last_read = w->key.key;
		pr_debug("%s", pkey(&w->key));

		closure_bio_submit(&w->io->bio, &w->io->cl, d->c->bio_split);
		atomic_inc(&d->count);
		if (atomic_inc_return(&d->in_flight) >= 8)
			goto out;

		spin_lock(&d->lock);
	}

	spin_unlock(&d->lock);
out:
	cached_dev_put(d);
}

static void read_dirty_work(struct work_struct *work)
{
	struct cached_dev *d = container_of(work, struct cached_dev, refill);
	spin_lock(&d->lock);
	read_dirty(d);
}

/* Insert data into cache */

static void put_data_bucket(struct open_bucket *b, struct cache_set *c,
			    struct bkey *k, struct bio *bio)
{
	unsigned split = min(bio_sectors(bio), b->sectors_free);

	for (int i = 0; i < KEY_PTRS(&b->key); i++)
		split = min(split, __bio_max_sectors(bio,
				      PTR_CACHE(c, &b->key, i)->bdev,
				      PTR_OFFSET(&b->key, i)));

	b->key.key += split;

	bkey_copy(k, &b->key);
	k->header += split << 20;

	b->sectors_free	-= split;

	/* If we're closing this open bucket, get_data_bucket()'s refcount now
	 * belongs to the key that's being inserted
	 */
	if (b->sectors_free < c->sb.block_size)
		b->sectors_free = 0;
	else
		for (unsigned i = 0; i < KEY_PTRS(&b->key); i++)
			atomic_inc(&PTR_BUCKET(c, &b->key, i)->pin);

	for (unsigned i = 0; i < KEY_PTRS(&b->key); i++) {
		atomic_long_add(split,
				&PTR_CACHE(c, &b->key, i)->sectors_written);

		b->key.ptr[i] += PTR(0, split, 0);
	}

	spin_unlock(&c->open_bucket_lock);
}

static struct open_bucket *get_data_bucket(struct bkey *search,
					   struct search *s)
{
	struct closure cl;
	struct closure *w = (s->insert_type == INSERT_WRITEBACK) ? &cl : NULL;
	struct cache_set *c = s->d->c;
	struct list_head *buckets = w ? &c->dirty_buckets : &c->open_buckets;
	struct open_bucket *l, *ret, *ret_task;
	__closure_init(&cl, NULL, true);
again:
	ret = ret_task = NULL;

	spin_lock(&c->open_bucket_lock);
	list_for_each_entry_reverse(l, buckets, list)
		if (!bkey_cmp(&l->key, search)) {
			ret = l;
			goto found;
		} else if (l->last == s->task)
			ret_task = l;

	ret = ret_task ?: list_first_entry(buckets, struct open_bucket, list);
found:
	if (!ret->sectors_free) {
		if (pop_bucket_set(c, initial_prio, &ret->key, 1, w)) {
			spin_unlock(&c->open_bucket_lock);
			if (!w)
				return NULL;
			closure_sync(w);
			goto again;
		}

		if (w)
			for (unsigned i = 0; i < KEY_PTRS(&ret->key); i++)
				PTR_BUCKET(c, &ret->key, i)->mark =
					GC_MARK_DIRTY;

		ret->sectors_free = c->sb.bucket_size;
	} else
		for (unsigned i = 0; i < KEY_PTRS(&ret->key); i++)
			BUG_ON(PTR_GEN(&ret->key, i) !=
			       PTR_BUCKET(s->d->c, &ret->key, i)->gen);

	ret->last = s->task;
	bkey_copy_key(&ret->key, search);

	list_move_tail(&ret->list, buckets);
	return ret;
}

static void bio_insert_endio(struct bio *bio, int error)
{
	struct search *s = bio->bi_private;

	if (error) {
		cache_set_error(s->d->c, "writing data to cache");
		if (s->insert_type == INSERT_WRITEBACK)
			s->error = error;

		if (s->insert_type != INSERT_WRITE)
			s->insert.fn = NULL;
		else {
			/* XXX: technically racy, if bio_insert() split and
			 * there was more than one error - should set
			 * s->insert.fn to something else
			 */
			for (struct bkey *k = s->keys.bottom;
			     k < s->keys.top;
			     k = next(k)) {
				void *p = next(k);
				size_t len = p - (void *) s->keys.top;

				k->header = KEY_HEADER(KEY_SIZE(k), KEY_DEV(k));
				memmove(next(k), p, len);
			}
		}
	}

	bio_put(bio);
	closure_put(&s->insert, delayed);
}

static void bio_insert(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, insert);
	struct bio *bio = s->cache_bio, *n;
	unsigned sectors = bio_sectors(bio);

	bio->bi_end_io	= bio_insert_endio;
	bio->bi_private = s;
	bio_get(bio);

	if (atomic_sub_return(bio_sectors(bio), &s->d->c->sectors_to_gc) < 0) {
		set_gc_sectors(s->d->c);
		queue_work(delayed, &s->d->c->gc_work);
	}

	do {
		struct open_bucket *b;
		struct bkey *k;

		if (keylist_realloc(&s->keys, 1))
			return_f(&s->insert, btree_journal);

		k = s->keys.top;

		b = get_data_bucket(&KEY(s->d->id, bio->bi_sector, 0), s);
		if (!b)
			goto err;

		put_data_bucket(b, s->d->c, k, bio);

		if (s->insert_type == INSERT_WRITEBACK)
			SET_KEY_DIRTY(k, true);

		n = bio_split_c(bio, KEY_SIZE(k), s->d->c);
		if (!n) {
			__bkey_put(s->d->c, k);
			return_f(&s->insert, bio_insert);
		}

		if (n != bio) {
			closure_get(&s->insert);
			n->bi_rw &= ~REQ_UNPLUG;
		}

		pr_debug("%s", pkey(k));
		keylist_push(&s->keys);

		s->insert.fn = btree_journal;

		n->bi_rw	|= WRITE;
		n->bi_sector	 = PTR_OFFSET(k, 0);
		n->bi_bdev	 = PTR_CACHE(s->d->c, k, 0)->bdev;

		if (n == bio) {
			maybe_refill_dirty(s);
			s->bio_done = true;
		}

		generic_make_request(n);
	} while (n != bio);

	return;
err:
	switch (s->insert_type) {
	case INSERT_WRITEBACK:
		BUG();
		/* Lookup in in_writeback rb tree, wait on appropriate
		 * closure, then invalidate in btree and do normal
		 * write
		 */
		s->bio_done = true;
		/* will call bio_insert_endio()... */
		bio_endio(s->bio, -ENOMEM);
		break;
	case INSERT_WRITE:
		s->skip = true;
		btree_invalidate(s);
		bio_endio(s->cache_bio, 0);
		break;
	case INSERT_READ:
		s->bio_done = true;
		bio_endio(s->cache_bio, 0);
	}

	pr_debug("error for %s, %i/%i sectors done, bi_sector %llu",
		 insert_type(s), sectors - bio_sectors(bio), sectors,
		 (uint64_t) bio->bi_sector);
}

/* Process a bio */

static void __bio_complete(struct search *s)
{
	if (s->bio && s->bi_end_io) {
		if (s->error)
			clear_bit(BIO_UPTODATE, &s->bio->bi_flags);

		s->bio->bi_private = s->bi_private;
		s->bio->bi_end_io = s->bi_end_io;
		s->bi_end_io(s->bio, s->error);
	}
	s->bio = NULL;
}

static void bio_complete(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, cl);
	struct cached_dev *d = s->d;

	BUG_ON(!keylist_empty(&s->keys));
	keylist_free(&s->keys);

	if (s->insert_type & INSERT_WRITE)
		up_read_non_owner(&d->writeback_lock);

	if (s->cache_bio) {
		int i;
		struct bio_vec *bv;

		__bio_for_each_segment(bv, s->cache_bio, i, s->pages_from)
			__free_page(bv->bv_page);
		bio_put(s->cache_bio);
	}

	if (s->error)
		pr_debug("error %i", s->error);

	__bio_complete(s);

	closure_del(&s->cl);
	mempool_free(s, d->search);
	cached_dev_put(d);
}

static void cache_request(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, cl);
	struct bkey *k;

	while ((k = keylist_pop(&s->keys)))
		__bkey_put(s->d->c, k);

	if (!s->lookup_done) {
		closure_init(&s->insert, &s->cl);
		return_f(&s->insert, __request_read);
	}

	if (s->cache_bio && !s->error && !atomic_read(&s->d->c->closing)) {
		struct bio_vec *bv = bio_iovec_idx(s->cache_bio, s->pages_from);

		while (s->pages_from) {
			struct page *p = alloc_page(__GFP_NOWARN|GFP_NOIO);
			if (!p)
				goto insert;

			s->pages_from--, bv--;

			memcpy(page_address(p), kmap(bv->bv_page), PAGE_SIZE);
			kunmap(bv->bv_page);
			bv->bv_page = p;
		}

		__bio_complete(s);
insert:
		closure_init(&s->insert, &s->cl);
		bio_insert(&s->insert);
	}

	return_f(cl, bio_complete);
}

static void request_endio(struct bio *bio, int error)
{
	struct search *s = bio->bi_private;
	s->error = error;
	bio_put(bio);
	closure_put(&s->cl, delayed);
}

static void cache_request_endio(struct bio *bio, int error)
{
	struct search *s = bio->bi_private;
	cache_set_err_on(error, s->d->c, "reading from cache");

	request_endio(bio, error);
}

static void check_should_skip(struct search *s)
{
	struct hlist_head *iohash(uint64_t k)
	{ return &s->d->io_hash[hash_64(k, RECENT_IO_BITS)]; }

	struct io t = { .sequential = 0 }, *i = &t;
	unsigned avg;

	if (atomic_read(&s->d->closing) ||
	    in_use(s->d->c) > CUTOFF_CACHE_ADD)
		goto skip;

	if (s->bio->bi_sector   % s->d->c->sb.block_size ||
	    bio_sectors(s->bio) % s->d->c->sb.block_size) {
		pr_debug("skipping unaligned io");
		goto skip;
	}

	if (!s->d->sequential_cutoff &&
	    !s->d->sequential_cutoff_average)
		goto rescale;

	if (s->insert_type & INSERT_WRITE &&
	    (s->d->writeback && (s->bio->bi_rw & REQ_SYNC)))
		goto rescale;

	spin_lock(&s->d->lock);

	if (s->d->sequential_merge) {
		struct hlist_node *cursor;

		hlist_for_each_entry(i, cursor, iohash(s->bio->bi_sector), hash)
			if (i->last == s->bio->bi_sector &&
			    time_before(jiffies, i->jiffies))
				goto found;

		i = list_first_entry(&s->d->io_lru, struct io, lru);
		i->sequential = 0;
		s->task->nr_ios++;
found:
		i->last = bio_end(s->bio);

		hlist_del(&i->hash);
		hlist_add_head(&i->hash, iohash(i->last));
		list_move_tail(&i->lru, &s->d->io_lru);
	} else
		s->task->nr_ios++;

	i->jiffies = jiffies + msecs_to_jiffies(5000);

	if (s->d->sequential_cutoff_average) {
		s->task->sequential_io += bio_sectors(s->bio);

		avg = s->task->sequential_io / (s->task->nr_ios + 1);
		if (avg > s->d->sequential_cutoff_average >> 9)
			goto skip_unlock;
	}

	if (s->d->sequential_cutoff) {
		i->sequential += bio_sectors(s->bio);

		if (i->sequential >= s->d->sequential_cutoff >> 9)
			goto skip_unlock;
	}

	spin_unlock(&s->d->lock);
rescale:
	rescale_heap(s->d->c, bio_sectors(s->bio));
	return;
skip_unlock:
	spin_unlock(&s->d->lock);
skip:
	atomic_long_add(bio_sectors(s->bio), &s->d->sectors_bypassed);
	s->skip = true;
}

static struct search *do_bio_hook(struct bio *bio, struct cached_dev *d)
{
	struct search *s = mempool_alloc(d->search, GFP_NOIO);
	search_init(s);
	closure_init(&s->cl, NULL);
	closure_init(&s->insert, &s->cl);

	s->d		= d;
	s->task		= get_current();
	s->bio		= bio;
	s->bi_end_io	= bio->bi_end_io;
	s->bi_private	= bio->bi_private;
	s->pages_from	= USHRT_MAX;
	s->cl.fn	= bio_complete;

	bio->bi_end_io	= request_endio;
	bio->bi_private = s;
	bio_get(bio);

	return s;
}

static void do_readahead(struct search *s, struct bio *last_bio, int sectors)
{
	struct bio *bio = NULL;
	int pages = bio_get_nr_vecs(last_bio->bi_bdev) * PAGE_SECTORS;

	if (sectors < 0 ||
	    bio_rw_flagged(last_bio, BIO_RW_AHEAD) ||
	    in_use(s->d->c) > CUTOFF_CACHE_READA)
		sectors = 0;
	else
		sectors = min(sectors, pages);

	pages = DIV_ROUND_UP(sectors, PAGE_SECTORS);
	s->cache_bio = bio_kmalloc(GFP_NOIO, last_bio->bi_max_vecs + pages);
	if (!s->cache_bio)
		return;

	__bio_clone(s->cache_bio, last_bio);
	s->pages_from = s->cache_bio->bi_vcnt;

	if (pages)
		bio = bio_kmalloc(GFP_NOIO, pages);
	if (!bio)
		return;

	bio->bi_sector	= bio_end(last_bio);
	bio->bi_bdev	= last_bio->bi_bdev;
	bio->bi_rw	= last_bio->bi_rw;
	bio->bi_size	= sectors << 9;

	/* XXX: don't want to pass an error all the way up, just need to make
	 * sure bio_insert doesn't get called */
	bio->bi_end_io	= request_endio;
	bio->bi_private	= s;

	bio_map(bio, NULL);
	if (bio_alloc_pages(bio, __GFP_NOWARN|GFP_NOIO)) {
		bio_put(bio);
		return;
	}

	memcpy(s->cache_bio->bi_io_vec + s->cache_bio->bi_vcnt,
	       bio->bi_io_vec, bio->bi_max_vecs * sizeof(struct bio_vec));

	s->cache_bio->bi_vcnt += bio->bi_vcnt;
	s->cache_bio->bi_size += bio->bi_size;

	pr_debug("%i + %i --> %i", bio_sectors(last_bio),
		 bio_sectors(bio), bio_sectors(s->cache_bio));
	atomic_long_inc(&s->d->cache_readaheads);

	closure_get(&s->cl);
	submit_bio(READ, bio);
}

static void __request_read(struct closure *cl)
{
	struct search *s = container_of(cl, struct search, insert);
	uint64_t reada = s->bio->bi_bdev->bd_inode->i_size;

	int ret = btree_root(search_recurse, s->d->c, s, &reada);

	if (ret == -ENOMEM) {
		closure_put(&s->cl, NULL);
		return_f(cl, NULL);
	}

	if (ret == -EAGAIN)
		return_f(cl, __request_read);

	s->lookup_done = true;

	if (!s->cache_hit) {
		reada = min_t(uint64_t, reada,
			      s->bio->bi_sector + (s->d->readahead >> 9));

		if (!s->skip)
			do_readahead(s, s->bio, reada - bio_end(s->bio));

		BUG_ON(s->cache_bio && !s->cache_bio->bi_size);

		generic_make_request(s->bio);
		atomic_long_inc(&s->d->cache_misses);
	} else
		atomic_long_inc(&s->d->cache_hits);

	return_f(cl, NULL);
}

static int request_read(struct search *s)
{
	s->cl.fn	= cache_request;
	s->insert_type	= INSERT_READ;
	check_should_skip(s);

	__request_read(&s->insert);
	return 0;
}

static int request_write(struct search *s)
{
	down_read_non_owner(&s->d->writeback_lock);
	s->insert_type = INSERT_WRITE;
	check_should_skip(s);

	if (in_writeback(s->d, s->bio->bi_sector, bio_sectors(s->bio))) {
		s->skip = false;
		s->insert_type = INSERT_WRITEBACK;
	}

	if (s->skip) {
skip:		s->cache_bio = bio_alloc_bioset(GFP_NOIO, 0,
						s->d->c->bio_split);
		s->cache_bio->bi_sector = s->bio->bi_sector;
		s->cache_bio->bi_size	= s->bio->bi_size;
		s->cache_bio->bi_flags |= 1 << BIO_HAS_POOL;
		s->cache_bio->bi_destructor = (void *) s->d->c->bio_split;

		btree_invalidate(s);
		closure_put(&s->insert, delayed);
		return 1;
	}

	if (s->d->writeback) {
		int i = in_use(s->d->c);
		if (i < CUTOFF_WRITEBACK ||
		    (i < CUTOFF_WRITEBACK_SYNC && s->bio->bi_rw & REQ_SYNC))
			s->insert_type = INSERT_WRITEBACK;
	}

	if (s->insert_type == INSERT_WRITE) {
		s->cache_bio = bio_kmalloc(GFP_NOIO, s->bio->bi_max_vecs);
		if (!s->cache_bio) {
			s->skip = true;
			goto skip;
		}

		__bio_clone(s->cache_bio, s->bio);
		bio_insert(&s->insert);
		return 1;
	} else {
		s->cache_bio = s->bio;
		closure_put(&s->cl, delayed);
		bio_insert(&s->insert);
		return 0;
	}
}

/* Sysfs */

#define write_attribute(n)	\
	static struct attribute sysfs_##n = { .name = #n, .mode = S_IWUSR }
#define read_attribute(n)	\
	static struct attribute sysfs_##n = { .name = #n, .mode = S_IRUSR }
#define rw_attribute(n)	\
	static struct attribute sysfs_##n =				\
		{ .name = #n, .mode = S_IWUSR|S_IRUSR }

#define sysfs_print(file, fmt, ...)					\
	if (attr == &sysfs_ ## file)					\
		return snprintf(buf, PAGE_SIZE, fmt "\n", __VA_ARGS__)

#define sysfs_hprint(file, val)						\
	if (attr == &sysfs_ ## file) {					\
		ssize_t ret = hprint(buf, val);				\
		strcat(buf, "\n");					\
		return ret + 1;						\
	}

#define sysfs_atoi(file, var)						\
	if (attr == &sysfs_ ## file) {					\
		unsigned long _v;					\
		int _r = strict_strtoul(buffer, 10, &_v);		\
		if (_r)							\
			return _r;					\
		var = _v;						\
	}

#define sysfs_hatoi(file, var)						\
	if (attr == &sysfs_ ## file)					\
		return strtoi_h(buffer, &var) ? : size;			\

static ssize_t show(struct kobject *, struct kobj_attribute *, char *);
static ssize_t store(struct kobject *, struct kobj_attribute *,
		     const char *, size_t);

static struct kobj_attribute sysfs_register =
	__ATTR(register, S_IWUSR, show, store);

static struct kobj_attribute sysfs_latency_warn_ms =
	__ATTR(latency_warn_ms, S_IWUSR|S_IRUSR, show, store);

write_attribute(attach);
write_attribute(detach);
write_attribute(unregister);
write_attribute(clear_stats);

read_attribute(bucket_size);
read_attribute(block_size);
read_attribute(nbuckets);
read_attribute(cache_hits);
read_attribute(cache_hit_ratio);
read_attribute(cache_misses);
read_attribute(cache_readaheads);
read_attribute(cache_miss_collisions);
read_attribute(tree_depth);
read_attribute(root_usage_percent);
read_attribute(priority_stats);
read_attribute(btree_cache_size);
read_attribute(heap_size);
read_attribute(written);
read_attribute(btree_written);
read_attribute(metadata_written);
read_attribute(bypassed);
read_attribute(btree_avg_keys_written);
read_attribute(active_journal_entries);

read_attribute(average_seconds_between_gc);
read_attribute(gc_ms_max);
read_attribute(seconds_since_gc);
read_attribute(btree_nodes);
read_attribute(btree_used_percent);
read_attribute(average_key_size);
read_attribute(dirty_data);

read_attribute(state);
read_attribute(writeback_keys_done);
read_attribute(writeback_keys_failed);

rw_attribute(sequential_cutoff);
rw_attribute(sequential_cutoff_average);
rw_attribute(sequential_merge);
rw_attribute(writeback);
rw_attribute(writeback_running);
rw_attribute(synchronous);
rw_attribute(discard);
rw_attribute(writeback_delay);
rw_attribute(writeback_percent);
rw_attribute(running);
rw_attribute(label);
rw_attribute(readahead);

static int btree_check(struct btree *b, struct search *s)
{
	struct bkey *k;

	for_each_key_filter(b, k, ptr_invalid) {
		for (unsigned i = 0; i < KEY_PTRS(k); i++) {
			struct bucket *g = PTR_BUCKET(b->c, k, i);

			if (!ptr_stale(b->c, k, i)) {
				g->gen = PTR_GEN(k, i);

				if (b->level)
					g->prio = btree_prio;
				else if (g->prio == btree_prio)
					g->prio = initial_prio;
			} else
				cache_bug_on(KEY_DIRTY(k) && KEY_SIZE(k),
					     b, "stale dirty pointer");
		}

		btree_mark_key(b, k);
	}

	if (b->level)
		for_each_key_filter(b, k, ptr_bad) {
			int ret = btree(check, b, k, s);
			if (ret)
				return ret;
		}

	return 0;
}

static struct dentry *debug;

#ifdef CONFIG_DEBUG_FS
static int btree_dump(struct btree *b, struct search *s, struct seq_file *f,
		      const char *tabs, uint64_t *prev, uint64_t *sectors)
{
	struct bkey *k;
	char buf[30];
	uint64_t last, biggest = 0;

	for_each_key(b, k) {
		int j = (uint64_t *) k - (*_i)->d;
		if (!j)
			last = *prev;

		if (last > k->key)
			seq_printf(f, "Key skipped backwards\n");

		if (!b->level && j &&
		    last != KEY_START(k))
			seq_printf(f, "<hole>\n");
		else if (b->level && !ptr_bad(b, k))
			btree(dump, b, k, s, f, tabs - 1, &last, sectors);

		seq_printf(f, "%s%li %4i: %s %s\n",
			   tabs, _i - b->sets, j, pkey(k), buf);

		if (!b->level && !buf[0])
			*sectors += KEY_SIZE(k);

		last = k->key;
		biggest = max(biggest, last);
	}
	*prev = biggest;

	return 0;
}

static int debug_seq_show(struct seq_file *f, void *data)
{
	static const char *tabs = "\t\t\t\t\t";
	uint64_t last = 0, sectors = 0;
	struct cache *ca = f->private;
	struct cache_set *c = ca->set;
	struct search s;

	search_init_stack(&s);

	btree_root(dump, c, &s, f, &tabs[4], &last, &sectors);

	seq_printf(f, "%s\n" "%llu Mb found\n",
		   pkey(&c->root->key), sectors / 2048);

	closure_sync(&s.insert);
	return 0;
}

static int debug_seq_open(struct inode *inode, struct file *file)
{
	return single_open(file, debug_seq_show, inode->i_private);
}

static const struct file_operations cache_debug_ops = {
	.owner		= THIS_MODULE,
	.open		= debug_seq_open,
	.read		= seq_read,
	.release	= single_release
};
#endif

/* Superblock/other metadata */

static const char *read_super(struct cache_sb *sb, struct block_device *bdev,
			      struct page **res)
{
	const char *err;
	struct cache_sb *s;
	struct buffer_head *bh = __bread(bdev, 1, SB_SIZE);

	if (!bh)
		return "IO error";

	s = (struct cache_sb *) bh->b_data;

	sb->offset		= le64_to_cpu(s->offset);
	sb->version		= le64_to_cpu(s->version);

	memcpy(sb->magic,	s->magic, 16);
	memcpy(sb->uuid,	s->uuid, 16);
	memcpy(sb->set_uuid,	s->set_uuid, 16);
	memcpy(sb->label,	s->label, SB_LABEL_SIZE);

	sb->flags		= le64_to_cpu(s->flags);
	sb->seq			= le64_to_cpu(s->seq);

	sb->nbuckets		= le64_to_cpu(s->nbuckets);
	sb->block_size		= le16_to_cpu(s->block_size);
	sb->bucket_size		= le16_to_cpu(s->bucket_size);

	sb->nr_in_set		= le16_to_cpu(s->nr_in_set);
	sb->nr_this_dev		= le16_to_cpu(s->nr_this_dev);
	sb->last_mount		= le32_to_cpu(s->last_mount);

	sb->first_bucket	= le16_to_cpu(s->first_bucket);
	sb->keys		= le16_to_cpu(s->keys);

	for (int i = 0; i < SB_JOURNAL_BUCKETS; i++)
		sb->d[i] = le64_to_cpu(s->d[i]);

	pr_debug("read sb version %llu, flags %llu, seq %llu, journal size %u",
		 sb->version, sb->flags, sb->seq, sb->keys);

	err = "Not a bcache superblock";
	if (sb->offset != SB_SECTOR)
		goto err;

	if (memcmp(sb->magic, bcache_magic, 16))
		goto err;

	err = "Too many journal buckets";
	if (sb->keys > SB_JOURNAL_BUCKETS)
		goto err;

	err = "Bad checksum";
	if (s->csum != csum_set(s))
		goto err;

	err = "Bad UUID";
	if (is_zero(sb->uuid, 16))
		goto err;

	err = "Unsupported superblock version";
	if (sb->version > CACHE_BACKING_DEV)
		goto err;

	err = "Bad block/bucket size";
	if (!is_power_of_2(sb->block_size) || sb->block_size > PAGE_SECTORS ||
	    !is_power_of_2(sb->bucket_size) || sb->bucket_size < PAGE_SECTORS)
		goto err;

	err = "Too many buckets";
	if (sb->nbuckets > LONG_MAX)
		goto err;

	err = "Not enough buckets";
	if (sb->nbuckets < 1 << 7)
		goto err;

	err = "Invalid superblock: device too small";
	if (get_capacity(bdev->bd_disk) < sb->bucket_size * sb->nbuckets)
		goto err;

	if (sb->version == CACHE_BACKING_DEV)
		goto out;

	err = "Bad UUID";
	if (is_zero(sb->set_uuid, 16))
		goto err;

	err = "Bad cache device number in set";
	if (!sb->nr_in_set ||
	    sb->nr_in_set <= sb->nr_this_dev ||
	    sb->nr_in_set > MAX_CACHES_PER_SET)
		goto err;

	err = "Journal buckets not sequential";
	for (int i = 0; i < sb->keys; i++)
		if (sb->d[i] != sb->first_bucket + i)
			goto err;

	err = "Too many journal buckets";
	if (sb->first_bucket + sb->keys > sb->nbuckets)
		goto err;

	err = "Invalid superblock: first bucket comes before end of super";
	if (sb->first_bucket * sb->bucket_size < 16)
		goto err;
out:
	sb->last_mount = get_seconds();
	err = NULL;

	get_page(bh->b_page);
	*res = bh->b_page;
err:
	put_bh(bh);
	return err;
}

static void write_bdev_super_endio(struct bio *bio, int error)
{
	struct cached_dev *d = bio->bi_private;
	/* XXX: error checking */

	if (d->sb_writer)
		closure_put(d->sb_writer, delayed);
	d->sb_writer = NULL;
	up(&d->sb_write);
}

static void __write_super(struct cache_sb *sb, struct bio *bio)
{
	struct cache_sb *out = page_address(bio->bi_io_vec[0].bv_page);

	bio->bi_sector	= SB_SECTOR;
	bio->bi_rw	= WRITE_SYNC|REQ_META;
	bio->bi_size	= SB_SIZE;
	bio_map(bio, NULL);

	out->offset		= cpu_to_le64(sb->offset);
	out->version		= cpu_to_le64(sb->version);

	memcpy(out->uuid,	sb->uuid, 16);
	memcpy(out->set_uuid,	sb->set_uuid, 16);
	memcpy(out->label,	sb->label, SB_LABEL_SIZE);

	out->flags		= cpu_to_le64(sb->flags);
	out->seq		= cpu_to_le64(sb->seq);

	out->last_mount		= cpu_to_le32(sb->last_mount);
	out->first_bucket	= cpu_to_le16(sb->first_bucket);
	out->keys		= cpu_to_le16(sb->keys);

	for (int i = 0; i < sb->keys; i++)
		out->d[i] = cpu_to_le64(sb->d[i]);

	out->csum = csum_set(out);

	pr_debug("ver %llu, flags %llu, seq %llu",
		 sb->version, sb->flags, sb->seq);

	submit_bio(WRITE, bio);
}

static void write_bdev_super(struct cached_dev *d, struct closure *cl)
{
	struct bio *bio = &d->sb_bio;

	down(&d->sb_write);

	bio_reset(bio);
	bio->bi_bdev	= d->bdev;
	bio->bi_end_io	= write_bdev_super_endio;
	bio->bi_private = d;

	if (cl)
		closure_get(cl);
	d->sb_writer = cl;

	__write_super(&d->sb, bio);

	if (cl)
		closure_sync(cl);
}

static void write_super_endio(struct bio *bio, int error)
{
	struct cache *c = bio->bi_private;

	cache_err_on(error, c, "writing superblock");
	closure_put(c->set->sb_writer, delayed);
}

static void write_super(struct cache_set *c, struct closure *cl)
{
	struct cache *ca;

	mutex_lock(&c->sb_write);
	c->sb.seq++;
	c->sb_writer = cl;

	for_each_cache(ca, c) {
		struct bio *bio = &ca->sb_bio;

		ca->sb.version		= c->sb.version;
		ca->sb.flags		= c->sb.flags;
		ca->sb.seq		= c->sb.seq;
		ca->sb.last_mount	= c->sb.last_mount;

		bio_reset(bio);
		bio->bi_bdev	= ca->bdev;
		bio->bi_end_io	= write_super_endio;
		bio->bi_private = ca;

		closure_get(cl);
		__write_super(&ca->sb, bio);
	}

	closure_sync(cl);
	mutex_unlock(&c->sb_write);
}

/* Bucket priorities/gens */

static void prio_endio(struct bio *bio, int error)
{
	struct cache *c = bio->bi_private;
	cache_err_on(error, c, "writing priorities");

	bio_put(bio);
	closure_put(&c->prio, system_wq);
}

static void prio_io(struct cache *c, uint64_t bucket, unsigned long rw)
{
	struct bio *bio = c->prio_bio;

	bio_reset(bio);
	bio->bi_sector	= bucket * c->sb.bucket_size;
	bio->bi_bdev	= c->bdev;
	bio->bi_rw	= REQ_META|rw;
	bio->bi_size	= bucket_bytes(c);

	bio->bi_end_io	= prio_endio;
	bio->bi_private = c;
	bio_map(bio, c->disk_buckets);

	closure_bio_submit(bio, &c->prio, c->set ? c->set->bio_split : NULL);
}

static void prio_write_done(struct closure *cl)
{
	struct cache *c = container_of(cl, struct cache, prio);
	pr_debug("");

	spin_lock(&c->set->bucket_lock);

	for (int i = 0; i < prio_buckets(c); i++)
		c->prio_buckets[i] = c->prio_next[i];

	c->prio_alloc = 0;
	c->need_save_prio = 0;

	spin_unlock(&c->set->bucket_lock);

	atomic_set(&c->prio_written, 1);
	closure_run_wait(&c->set->bucket_wait, delayed);

	return_f(cl, NULL);
}

static void prio_write_journal(struct closure *cl)
{
	struct cache *c = container_of(cl, struct cache, prio);

	c->prio_start = c->prio_next[0];
	btree_journal_meta(c->set, cl);

	return_f(cl, prio_write_done);
}

static void prio_write_bucket(struct closure *cl)
{
	struct cache *c = container_of(cl, struct cache, prio);
	struct prio_set *p = c->disk_buckets;
	struct bucket_disk *d = p->data, *end = d + prios_per_bucket(c);

	int i = c->prio_write++;

	if (c->prio_write != prio_buckets(c))
		p->next_bucket = c->prio_next[c->prio_write];
	else
		cl->fn = prio_write_journal;

	for (struct bucket *b = c->buckets + i * prios_per_bucket(c);
	     b < c->buckets + c->sb.nbuckets && d < end;
	     b++, d++) {
		d->prio = cpu_to_le16(b->prio);
		d->gen = b->disk_gen;
	}

	p->magic = pset_magic(c);
	p->csum = crc64(&p->magic, bucket_bytes(c) - 8);

	prio_io(c, c->prio_next[i], WRITE_SYNC);
}

static void prio_write(struct cache *c, struct closure *cl)
{
	BUG_ON(c->prio_alloc != prio_buckets(c));

	for (struct bucket *b = c->buckets;
	     b < c->buckets + c->sb.nbuckets; b++)
		b->disk_gen = b->gen;

	closure_init(&c->prio, cl);
	c->prio.fn = prio_write_bucket;

	c->prio_write = 0;
	c->disk_buckets->seq++;

	atomic_long_add(c->sb.bucket_size * prio_buckets(c),
			&c->meta_sectors_written);

	atomic_set(&c->prio_written, -1);
	closure_put(&c->prio, system_wq);

	pr_debug("starting prio write");
}

static int prio_read(struct cache *c, uint64_t bucket)
{
	struct prio_set *p = c->disk_buckets;
	struct bucket_disk *d = p->data + prios_per_bucket(c), *end = d;

	closure_init(&c->prio, NULL);

	for (struct bucket *b = c->buckets;
	     b < c->buckets + c->sb.nbuckets;
	     b++, d++) {
		if (d == end) {
			c->prio_buckets[c->prio_write++] = bucket;

			closure_get(&c->prio);
			prio_io(c, bucket, READ_SYNC);
			closure_sync(&c->prio);

			/* XXX: doesn't get error handling right with splits */
			if (!test_bit(BIO_UPTODATE, &c->prio_bio->bi_flags))
				return_f(&c->prio, NULL, -1);

			if (p->csum != crc64(&p->magic, bucket_bytes(c) - 8))
				printk(KERN_WARNING "bcache: "
				       "bad csum reading priorities\n");

			if (p->magic != pset_magic(c))
				printk(KERN_WARNING "bcache: "
				       "bad magic reading priorities\n");

			bucket = p->next_bucket;
			d = p->data;
		}

		b->prio = le16_to_cpu(d->prio);
		b->gen = b->disk_gen = b->last_gc = b->gc_gen = d->gen;
	}

	return_f(&c->prio, NULL, 0);
}

/* Backing device */

static void run_dev(struct cached_dev *d)
{
	if (atomic_xchg(&d->running, 1))
		return;

	if (!d->c &&
	    BDEV_STATE(&d->sb) != BDEV_STATE_NONE) {
		struct closure cl;
		closure_init_stack(&cl);

		SET_BDEV_STATE(&d->sb, BDEV_STATE_STALE);
		write_bdev_super(d, &cl);
	}

	add_disk(d->disk);
#if 0
	char *env[] = { "SYMLINK=label" , NULL };
	kobject_uevent_env(&disk_to_dev(d->disk)->kobj, KOBJ_CHANGE, env);
#endif
	if (sysfs_create_link(&d->kobj, &disk_to_dev(d->disk)->kobj, "dev") ||
	    sysfs_create_link(&disk_to_dev(d->disk)->kobj, &d->kobj, "bcache"))
		pr_debug("error creating sysfs link");
}

static void detach_dev(struct cached_dev *d)
{
	char buf[BDEVNAME_SIZE];
	struct closure cl;
	closure_init_stack(&cl);

	memset(&d->sb.set_uuid, 0, 16);
	SET_BDEV_STATE(&d->sb, BDEV_STATE_NONE);
	write_bdev_super(d, &cl);

	memcpy(d->c->uuids[d->id].uuid, invalid_uuid, 16);
	d->c->uuids[d->id].invalidated = cpu_to_le32(get_seconds());
	uuid_write(d->c);

	BUG_ON(!atomic_read(&d->closing));
	BUG_ON(atomic_read(&d->count));

	sprintf(buf, "bdev%i", d->id);

	sysfs_remove_link(&d->c->kobj, buf);
	sysfs_remove_link(&d->kobj, "cache");

	list_move(&d->list, &uncached_devices);
	atomic_set(&d->closing, 0);
	kobject_put(&d->c->kobj);
	d->c = NULL;

	printk(KERN_DEBUG "bcache: Caching disabled for %s\n",
	       bdevname(d->bdev, buf));
}

static inline void cached_dev_put(struct cached_dev *d)
{
	if (atomic_dec_and_test(&d->count)) {
		mutex_lock(&register_lock);
		detach_dev(d);
		mutex_unlock(&register_lock);
	}
}

static void cached_dev_close(struct cached_dev *d)
{
	if (atomic_xchg(&d->closing, 1))
		return;

	if (should_refill_dirty(d) &&
	    queue_work(writeback, &d->refill))
		return;

	if (atomic_dec_and_test(&d->count))
		detach_dev(d);
}

static int register_dev_on_set(struct cache_set *c, struct cached_dev *d)
{
	uint32_t rtime = cpu_to_le32(get_seconds());
	struct uuid_entry *u;
	struct closure cl;
	const char *msg = "looked up";
	char buf[BDEVNAME_SIZE];
	bdevname(d->bdev, buf);

	closure_init_stack(&cl);
	if (d->c ||
	    atomic_read(&c->closing) ||
	    memcmp(d->sb.set_uuid, c->sb.set_uuid, 16))
		return -ENOENT;

	if (d->sb.block_size < c->sb.block_size) {
		err_printk("Couldn't attach %s: block size "
			   "less than set's block size\n", buf);
		return -EINVAL;
	}

	for (u = c->uuids; u < c->uuids + c->nr_uuids; u++)
		if (!memcmp(u->uuid, d->sb.uuid, 16)) {
			if (BDEV_STATE(&d->sb) != BDEV_STATE_STALE)
				goto found;

			memcpy(u->uuid, invalid_uuid, 16);
			u->invalidated = cpu_to_le32(get_seconds());
			break;
		}

	if (BDEV_STATE(&d->sb) == BDEV_STATE_DIRTY) {
		err_printk("Couldn't find uuid for %s in set\n", buf);
		return -ENOENT;
	}

	for (u = c->uuids; u < c->uuids + c->nr_uuids; u++)
		if (is_zero(u->uuid, 16))
			goto found;

	err_printk("Not caching %s, no room for UUID\n", buf);
	return -EINVAL;
found:
	sprintf(buf, "bdev%zi", u - c->uuids);
	if (sysfs_create_link(&d->kobj, &c->kobj, "cache") ||
	    sysfs_create_link(&c->kobj, &d->kobj, buf))
		return -ENOMEM;

	/* Deadlocks since we're called via sysfs...
	sysfs_remove_file(&d->kobj, &sysfs_attach);
	 */

	if (is_zero(u->uuid, 16)) {
		memcpy(u->uuid, d->sb.uuid, 16);
		memcpy(u->label, d->sb.label, SB_LABEL_SIZE);
		u->first_reg = u->last_reg = rtime;
		uuid_write(c);

		memcpy(d->sb.set_uuid, c->sb.set_uuid, 16);
		SET_BDEV_STATE(&d->sb, d->writeback
			     ? BDEV_STATE_DIRTY
			     : BDEV_STATE_CLEAN);
		write_bdev_super(d, &cl);

		msg = "inserted new";
	} else {
		u->last_reg = rtime;
		uuid_write(c);
	}

	d->id = u - c->uuids;
	d->c = c;
	kobject_get(&c->kobj);
	list_move(&d->list, &c->devices);

	smp_wmb();
	/* d->c must be set before d->count != 0 */
	atomic_set(&d->count, 1);

	if (BDEV_STATE(&d->sb) == BDEV_STATE_DIRTY)
		queue_writeback(d);

	run_dev(d);

	printk(KERN_INFO "bcache: Caching %s, %s UUID %pU\n",
	       bdevname(d->bdev, buf), msg, d->sb.uuid);
	return 0;
}

static ssize_t show_dev(struct kobject *kobj, struct attribute *attr, char *buf)
{
	struct cached_dev *d = container_of(kobj, struct cached_dev, kobj);
	const char *states[] = { "no cache", "clean", "dirty", "inconsistent" };

	unsigned cache_hit_ratio(struct cached_dev *d)
	{
		unsigned long hits = atomic_long_read(&d->cache_hits);
		unsigned long misses = atomic_long_read(&d->cache_misses);
		unsigned long total = hits + misses;
		return total ? hits * 100 / total : 0;
	}

	sysfs_print(writeback, "%i",		d->writeback);
	sysfs_print(writeback_running, "%i",	d->writeback_running);
	sysfs_print(writeback_delay, "%i",	d->writeback_delay);
	sysfs_print(writeback_percent, "%i",	d->writeback_percent);
	sysfs_hprint(sequential_cutoff,		d->sequential_cutoff);
	sysfs_hprint(sequential_cutoff_average,	d->sequential_cutoff_average);
	sysfs_hprint(sequential_merge,		d->sequential_merge);
	sysfs_hprint(bypassed, atomic_long_read(&d->sectors_bypassed) << 9);
	sysfs_print(cache_hits, "%lu", atomic_long_read(&d->cache_hits));
	sysfs_print(cache_misses, "%lu", atomic_long_read(&d->cache_misses));
	sysfs_print(cache_hit_ratio,	"%u", cache_hit_ratio(d));
	sysfs_print(cache_readaheads, "%lu",
		    atomic_long_read(&d->cache_readaheads));
	sysfs_print(cache_miss_collisions, "%lu",
		    atomic_long_read(&d->cache_miss_collisions));
	sysfs_print(running, "%i",		atomic_read(&d->running));
	sysfs_print(state, "%s",		states[BDEV_STATE(&d->sb)]);
	sysfs_hprint(readahead,                 d->readahead);

	if (attr == &sysfs_label) {
		memcpy(buf, d->sb.label, SB_LABEL_SIZE);
		buf[SB_LABEL_SIZE + 1] = '\0';
		strcat(buf, "\n");
		return strlen(buf);
	}

	return 0;
}

static ssize_t __store_dev(struct cached_dev *d, struct attribute *attr,
			   const char *buffer, size_t size)
{
	unsigned v = size;
	struct cache_set *c;
	struct closure cl;
	closure_init_stack(&cl);

	sysfs_atoi(writeback_running,	d->writeback_running);
	sysfs_atoi(writeback_delay,	d->writeback_delay);
	sysfs_atoi(sequential_merge,	d->sequential_merge);
	sysfs_hatoi(sequential_cutoff,	d->sequential_cutoff);
	sysfs_hatoi(sequential_cutoff_average,	d->sequential_cutoff_average);
	sysfs_hatoi(readahead,          d->readahead);

	if (attr == &sysfs_clear_stats) {
		atomic_long_set(&d->sectors_bypassed, 0);
		atomic_long_set(&d->cache_hits, 0);
		atomic_long_set(&d->cache_misses, 0);
		atomic_long_set(&d->cache_readaheads, 0);
		atomic_long_set(&d->cache_miss_collisions, 0);
	}

	if (attr == &sysfs_running) {
		sysfs_atoi(running, v);
		if (v)
			run_dev(d);
	}

	if (attr == &sysfs_writeback_percent) {
		sysfs_atoi(writeback_percent, v);
		d->writeback_percent = min(v, 40U);
	}

	if (attr == &sysfs_writeback) {
		sysfs_atoi(writeback, v);
		SET_BDEV_WRITEBACK(&d->sb, v);

		if (v &&
		    d->c &&
		    BDEV_STATE(&d->sb) == BDEV_STATE_CLEAN) {
			SET_BDEV_STATE(&d->sb, BDEV_STATE_DIRTY);
			write_bdev_super(d, &cl);
		} else
			write_bdev_super(d, NULL);

		d->writeback = v;
	}

	if (attr == &sysfs_label) {
		memcpy(d->sb.label, buffer, SB_LABEL_SIZE);
		write_bdev_super(d, NULL);
		if (d->c) {
			memcpy(d->c->uuids[d->id].label, buffer, SB_LABEL_SIZE);
			uuid_write(d->c);
		}
	}

	if (attr == &sysfs_attach) {
		if (parse_uuid(buffer, d->sb.set_uuid) < 16)
			return -EINVAL;

		list_for_each_entry(c, &cache_sets, list) {
			v = register_dev_on_set(c, d);
			if (!v)
				return size;
		}
		size = v;
	}

	if (attr == &sysfs_detach && d->c)
		cached_dev_close(d);

	return size;
}

static ssize_t store_dev(struct kobject *kobj, struct attribute *attr,
			 const char *buffer, size_t size)
{
	struct cached_dev *d = container_of(kobj, struct cached_dev, kobj);

	/* XXX: this looks sketchy as hell */
	if (attr == &sysfs_unregister &&
	    !atomic_xchg(&d->unregister, 1))
		kobject_put(&d->kobj);

	mutex_lock(&register_lock);
	size = __store_dev(d, attr, buffer, size);

	if ((attr == &sysfs_writeback_running ||
	     attr == &sysfs_writeback_percent ||
	     attr == &sysfs_writeback) &&
	    should_refill_dirty(d) &&
	    atomic_inc_not_zero(&d->count)) {
		mutex_unlock(&register_lock);
		read_dirty_work(&d->refill);
		return size;
	}

	mutex_unlock(&register_lock);
	return size;
}

static void free_dev(struct cached_dev *d)
{
	if (d) {
		mempool_destroy(d->search);
		list_del(&d->list);
		kfree(d);
	}
	module_put(THIS_MODULE);
}

static void unregister_dev_kobj(struct kobject *k)
{
	struct cached_dev *d = container_of(k, struct cached_dev, kobj);

	/* XXX: background writeback could be in progress... */
	cancel_work_sync(&d->refill);

	mutex_lock(&register_lock);

	if (d->c)
		kobject_put(&d->c->kobj);

	blkdev_put(d->bdev, FMODE_READ|FMODE_WRITE);
	free_dev(d);

	mutex_unlock(&register_lock);
}

static struct cached_dev *alloc_backing_dev(void)
{
	struct cached_dev *d = kzalloc(sizeof(struct cached_dev), GFP_KERNEL);
	if (!d)
		return NULL;

	d->search = mempool_create_slab_pool(32, search_cache);
	if (!d->search) {
		kfree(d);
		return NULL;
	}

	INIT_WORK(&d->refill, read_dirty_work);
	spin_lock_init(&d->lock);
	init_rwsem(&d->writeback_lock);
	sema_init(&d->sb_write, 1);

	d->dirty			= RB_ROOT;
	d->writeback_running		= true;
	d->writeback_delay		= 30;

	d->sequential_merge		= true;
	d->sequential_cutoff		= 4 << 20;
	d->sequential_cutoff_average	= 4 << 20;

	INIT_LIST_HEAD(&d->io_lru);
	d->sb_bio.bi_io_vec = d->sb_bio.bi_inline_vecs;

	for (struct io *j = d->io; j < d->io + RECENT_IO; j++) {
		list_add(&j->lru, &d->io_lru);
		hlist_add_head(&j->hash, d->io_hash + RECENT_IO);
	}

	return d;
}

static int register_dev_kobj(struct cached_dev *d)
{
	static struct attribute *files[] = {
		&sysfs_attach,
		&sysfs_detach,
		/* Not ready yet
		&sysfs_unregister,
		*/
		&sysfs_writeback,
		&sysfs_writeback_running,
		&sysfs_writeback_delay,
		&sysfs_writeback_percent,
		&sysfs_sequential_cutoff,
		&sysfs_sequential_cutoff_average,
		&sysfs_sequential_merge,
		&sysfs_bypassed,
		&sysfs_cache_hits,
		&sysfs_cache_misses,
		&sysfs_cache_hit_ratio,
		&sysfs_cache_readaheads,
		&sysfs_cache_miss_collisions,
		&sysfs_clear_stats,
		&sysfs_running,
		&sysfs_state,
		&sysfs_label,
		&sysfs_readahead,
		NULL
	};
	static const struct sysfs_ops ops = {
		.show = show_dev,
		.store = store_dev
	};
	static struct kobj_type dev_obj = {
		.release = unregister_dev_kobj,
		.sysfs_ops = &ops,
		.default_attrs = files
	};

	struct cache_set *c;
	struct kobject *p = &part_to_dev(d->bdev->bd_part)->kobj;
	int ret = kobject_init_and_add(&d->kobj, &dev_obj, p, "bcache");

	if (!ret) {
		list_add(&d->list, &uncached_devices);
		list_for_each_entry(c, &cache_sets, list)
			register_dev_on_set(c, d);
	}

	return ret;
}

/* Backing device - bcache superblock */

static int open_dev(struct block_device *b, fmode_t mode)
{
	struct cached_dev *d = b->bd_disk->private_data;
	kobject_get(&d->kobj);
	return 0;
}

static int release_dev(struct gendisk *b, fmode_t mode)
{
	struct cached_dev *d = b->private_data;
	kobject_put(&d->kobj);
	return 0;
}

static const struct block_device_operations bcache_ops = {
	.open		= open_dev,
	.release	= release_dev,
	.owner		= THIS_MODULE,
};

static int bcache_make_request(struct request_queue *q, struct bio *bio)
{
	struct search *s;
	struct cached_dev *d = bio->bi_bdev->bd_disk->private_data;

	bio->bi_bdev = d->bdev;
	bio->bi_sector += 16;

	if (!bio_has_data(bio) ||
	    !atomic_inc_not_zero(&d->count))
		return 1;

	s = do_bio_hook(bio, d);

	return bio->bi_rw & REQ_WRITE
		? request_write(s)
		: request_read(s);
}

static void bcache_unplug(struct request_queue *q)
{
	struct cached_dev *d = q->queuedata;

	blk_unplug(bdev_get_queue(d->bdev));

	if (atomic_inc_not_zero(&d->count)) {
		struct cache *c;

		for_each_cache(c, d->c)
			blk_unplug(bdev_get_queue(c->bdev));

		cached_dev_put(d);
	}
}

static const char *register_bdev(struct cache_sb *sb, struct page *sb_page,
				 struct block_device *bdev)
{
	char name[BDEVNAME_SIZE];
	const char *err = "cannot allocate memory";
	struct cached_dev *d = alloc_backing_dev();
	struct request_queue *q;
	if (!d)
		return err;

	memcpy(&d->sb, sb, sizeof(struct cache_sb));
	d->sb_bio.bi_io_vec[0].bv_page = sb_page;
	d->bdev = bdev;
	d->bdev->bd_holder = d;
	d->writeback = BDEV_WRITEBACK(&d->sb);

	d->disk = alloc_disk(1);
	if (!d->disk)
		goto err;

	snprintf(d->disk->disk_name, DISK_NAME_LEN, "bcache%i", bcache_minor);
	set_capacity(d->disk, d->bdev->bd_part->nr_sects - 16);

	d->disk->major		= bcache_major;
	d->disk->first_minor	= bcache_minor++;
	d->disk->fops		= &bcache_ops;
	d->disk->queue		= blk_alloc_queue(GFP_KERNEL);
	d->disk->private_data	= d;
	if (!d->disk->queue)
		goto err;

	blk_queue_make_request(d->disk->queue, bcache_make_request);

	q = bdev_get_queue(d->bdev);

	d->disk->queue->unplug_fn		= bcache_unplug;
	d->disk->queue->queuedata		= d;
	d->disk->queue->limits.max_hw_sectors	= q->limits.max_hw_sectors;
	d->disk->queue->limits.max_sectors	= q->limits.max_sectors;
	d->disk->queue->limits.max_segment_size	= q->limits.max_segment_size;
	d->disk->queue->limits.max_segments	= q->limits.max_segments;
	d->disk->queue->limits.logical_block_size  = block_bytes(d);
	d->disk->queue->limits.physical_block_size = block_bytes(d);
	set_bit(QUEUE_FLAG_NONROT, &d->disk->queue->queue_flags);

	err = "error creating kobject";
	if (register_dev_kobj(d))
		goto err;

	if (BDEV_STATE(&d->sb) == BDEV_STATE_NONE ||
	    BDEV_STATE(&d->sb) == BDEV_STATE_STALE)
		run_dev(d);

	return NULL;
err:
	printk(KERN_DEBUG "bcache: error opening %s: %s\n",
	       bdevname(bdev, name), err);
	free_dev(d);
	return NULL;
}

/* Cache set */

#define sum_bdev(stat)					\
size_t stat(struct cache_set *c)			\
{							\
	struct cached_dev *d;				\
	size_t ret = 0;					\
	list_for_each_entry(d, &c->devices, list)	\
		ret += atomic_long_read(&d->stat);	\
	return ret;					\
}

static ssize_t __show_cache_set(struct cache_set *c, struct attribute *attr,
				char *buf)
{
	sum_bdev(sectors_bypassed)
	sum_bdev(cache_hits)
	sum_bdev(cache_misses)
	sum_bdev(cache_readaheads)
	sum_bdev(cache_miss_collisions)

	unsigned cache_hit_ratio(struct cache_set *c)
	{
		unsigned long hits = cache_hits(c), misses = cache_misses(c);
		unsigned long total = hits + misses;
		return total ? hits * 100 / total : 0;
	}

	unsigned avg_keys_written(struct cache_set *c)
	{
		long writes = atomic_long_read(&c->btree_write_count);
		return writes
			? atomic_long_read(&c->keys_write_count) / writes
			: 0;
	}

	unsigned root_usage(struct cache_set *c)
	{
		unsigned bytes = 0;
		struct bkey *k;

		for_each_key_filter(c->root, k, ptr_bad)
			bytes += key_bytes(k);

		return (bytes * 100) / btree_bytes(c);
	}

	sysfs_print(synchronous,	"%i", (int) CACHE_SYNC(&c->sb));
	sysfs_hprint(bucket_size,	bucket_bytes(c));
	sysfs_hprint(block_size,	block_bytes(c));
	sysfs_print(tree_depth,		"%u", c->root->level);
	sysfs_print(root_usage_percent,	"%u", root_usage(c));
	sysfs_print(btree_avg_keys_written, "%u", avg_keys_written(c));
	sysfs_print(btree_cache_size,	"%i", btree_cache_size(c));

	sysfs_print(average_seconds_between_gc, "%li",
		    (get_seconds() - c->sb.last_mount) / c->gc_stats.count);
	sysfs_print(gc_ms_max,		"%u", c->gc_stats.ms_max);
	sysfs_print(seconds_since_gc,	"%li",
		    get_seconds() - c->gc_stats.last);
	sysfs_print(btree_nodes,	"%zu", c->gc_stats.nodes);
	sysfs_print(btree_used_percent,	"%u", btree_used(c));
	sysfs_hprint(average_key_size,	c->gc_stats.data / c->gc_stats.nkeys);
	sysfs_hprint(dirty_data,	c->gc_stats.dirty);

	sysfs_print(writeback_keys_done,	"%li",
		    atomic_long_read(&c->writeback_keys_done));
	sysfs_print(writeback_keys_failed,	"%li",
		    atomic_long_read(&c->writeback_keys_failed));
	sysfs_print(active_journal_entries,	"%zu",
		    fifo_used(&c->journal.pin));

	sysfs_hprint(bypassed,		sectors_bypassed(c) << 9);
	sysfs_print(cache_hits,		"%lu",	cache_hits(c));
	sysfs_print(cache_misses,	"%lu", cache_misses(c));
	sysfs_print(cache_hit_ratio,	"%u", cache_hit_ratio(c));
	sysfs_print(cache_readaheads,	"%lu", cache_readaheads(c));
	sysfs_print(cache_miss_collisions, "%lu", cache_miss_collisions(c));

	return 0;
}

static ssize_t show_cache_set(struct kobject *kobj, struct attribute *attr,
			      char *buf)
{
	struct cache_set *c = container_of(kobj, struct cache_set, kobj);
	ssize_t ret;

	mutex_lock(&register_lock);
	ret = __show_cache_set(c, attr, buf);
	mutex_unlock(&register_lock);
	return ret;
}

static ssize_t store_cache_set(struct kobject *kobj, struct attribute *attr,
			       const char *buffer, size_t size)
{
	struct cache_set *c = container_of(kobj, struct cache_set, kobj);
	struct closure cl;
	closure_init_stack(&cl);

	if (attr == &sysfs_unregister &&
	    !atomic_xchg(&c->closing, 1))
		schedule_work(&c->unregister);

	if (attr == &sysfs_synchronous) {
		bool sync;
		sysfs_atoi(synchronous, sync);

		if (sync != CACHE_SYNC(&c->sb)) {
			mutex_lock(&register_lock);

			SET_CACHE_SYNC(&c->sb, sync);

			write_super(c, &cl);

			mutex_unlock(&register_lock);
		}
	}

	if (attr == &sysfs_clear_stats) {
		atomic_long_set(&c->btree_write_count, 0);
		atomic_long_set(&c->keys_write_count, 0);
	}

	return size;
}

static bool cache_set_error(struct cache_set *c, const char *m, ...)
{
	va_list args;

	if (atomic_xchg(&c->closing, 1))
		return false;

	/* XXX: we can be called from atomic context
	acquire_console_sem();
	*/

	printk(KERN_ERR "bcache error on %pU: ", c->sb.set_uuid);

	va_start(args, m);
	vprintk(m, args);
	va_end(args);

	printk(", disabling caching\n");

	queue_work(delayed, &c->unregister);
	return true;
}

static void free_cache_set(struct cache_set *c)
{
	struct btree *b;
	struct open_bucket *o;

	if (c->shrink.list.next)
		unregister_shrinker(&c->shrink);

	list_splice(&c->lru, &c->freed);
	while (!list_empty(&c->freed)) {
		b = list_first_entry(&c->freed, struct btree, lru);
		list_del(&b->lru);
		cancel_delayed_work_sync(&b->work);
		free_pages((unsigned long) b->data, b->page_order);
		kfree(b);
	}

	list_splice(&c->dirty_buckets, &c->open_buckets);
	while (!list_empty(&c->open_buckets)) {
		o = list_first_entry(&c->open_buckets,
				     struct open_bucket, list);
		list_del(&o->list);
		kfree(o);
	}

	free_pages((unsigned long) c->journal.w[1].data, JSET_BITS);
	free_pages((unsigned long) c->journal.w[0].data, JSET_BITS);
	free_pages((unsigned long) c->uuids, ilog2(bucket_pages(c)));
	free_pages((unsigned long) c->sort, ilog2(bucket_pages(c)));

	kfree(c->fill_iter);
	bioset_free(c->bio_split);
	kfree(c);
}

static void unregister_cache_set(struct kobject *k)
{
	struct cache_set *c = container_of(k, struct cache_set, kobj);
	struct cache *ca;
	struct btree *b;
	struct search s;
	search_init_stack(&s);
	lockdep_assert_held(&register_lock);

	list_del(&c->list);

	if (!IS_ERR_OR_NULL(c->root))
		list_add(&c->root->lru, &c->lru);

	list_for_each_entry(b, &c->lru, lru)
		if (b->write)
			btree_write(b, true, &s);

	closure_sync(&s.insert);

	cancel_work_sync(&c->gc_work);
	cancel_work_sync(&c->journal.work);

	for_each_cache(ca, c)
		kobject_put(&ca->kobj);

	free_cache_set(c);
}

static void unregister_cache_set_work(struct work_struct *w)
{
	struct cache_set *c = container_of(w, struct cache_set, unregister);
	struct cached_dev *d;

	mutex_lock(&register_lock);

	list_for_each_entry(d, &c->devices, list)
		cached_dev_close(d);

	kobject_put(&c->kobj);

	mutex_unlock(&register_lock);
}

#define alloc_bucket_pages(gfp, c)			\
	((void *) __get_free_pages(__GFP_ZERO|gfp, ilog2(bucket_pages(c))))

static struct cache_set *alloc_cache_set(struct cache_sb *sb)
{
	int iter_size;
	struct cache_set *c;
	c = kzalloc(sizeof(struct cache_set) +
		    sizeof(struct hlist_head) * (1 << BUCKET_HASH_BITS),
		    GFP_KERNEL);
	if (!c)
		return NULL;

	memcpy(c->sb.set_uuid, sb->set_uuid, 16);
	c->sb.block_size	= sb->block_size;
	c->sb.bucket_size	= sb->bucket_size;
	c->sb.nr_in_set		= sb->nr_in_set;
	c->sb.last_mount	= sb->last_mount;
	c->bucket_bits		= ilog2(sb->bucket_size);
	c->nr_uuids		= bucket_bytes(c) / sizeof(struct uuid_entry);

	c->btree_pages		= c->sb.bucket_size / PAGE_SECTORS;
	if (c->btree_pages > BTREE_MAX_PAGES)
		c->btree_pages = max_t(int, c->btree_pages / 4,
				       BTREE_MAX_PAGES);

	spin_lock_init(&c->bucket_lock);
	spin_lock_init(&c->open_bucket_lock);
	mutex_init(&c->gc_lock);
	mutex_init(&c->fill_lock);
	mutex_init(&c->sort_lock);
	mutex_init(&c->sb_write);

	INIT_WORK(&c->unregister, unregister_cache_set_work);
	INIT_WORK(&c->gc_work, btree_gc_work);
	INIT_WORK(&c->journal.work, btree_journal_work);
	INIT_LIST_HEAD(&c->devices);
	INIT_LIST_HEAD(&c->lru);
	INIT_LIST_HEAD(&c->freed);
	INIT_LIST_HEAD(&c->open_buckets);
	INIT_LIST_HEAD(&c->dirty_buckets);

	atomic_set(&c->journal.io, -1);
	spin_lock_init(&c->journal.lock);
	c->journal.w[0].c = c;
	c->journal.w[1].c = c;

	iter_size = (sb->bucket_size / sb->block_size + 1) *
		sizeof(struct btree_iter_set);

	if (!(c->bio_split = bioset_create(64, 0)) ||
	    !(c->fill_iter = kmalloc(iter_size, GFP_KERNEL)) ||
	    !(c->sort = alloc_bucket_pages(GFP_KERNEL, c)) ||
	    !(c->uuids = alloc_bucket_pages(GFP_KERNEL, c)) ||
	    !(init_fifo(&c->journal.pin, 1023, GFP_KERNEL)) ||
	    !(c->journal.w[0].data = (void *) __get_free_pages(GFP_KERNEL,
							       JSET_BITS)) ||
	    !(c->journal.w[1].data = (void *) __get_free_pages(GFP_KERNEL,
							       JSET_BITS)))
		goto err;

	for (int i = 0; i < 16; i++) {
		struct open_bucket *b = kzalloc(sizeof(*b), GFP_KERNEL);
		if (!b)
			goto err;

		list_add(&b->list, i & 1
			 ? &c->open_buckets
			 : &c->dirty_buckets);
	}

	for (int i = 0; i < btree_reserve(c); i++) {
		struct btree *b = __alloc_bucket(c, GFP_KERNEL);
		if (!b)
			goto err;

		alloc_bucket_data(b);
		if (!b->data)
			goto err;

		list_move_tail(&b->lru, &c->lru);
	}

	c->shrink.shrink = shrink_buckets;
	c->shrink.seeks = 3;
	register_shrinker(&c->shrink);

	return c;
err:
	free_cache_set(c);
	return NULL;
}

static void run_cache_set(struct cache_set *c)
{
	void fill_heap(void)
	{
		struct bucket *b;
		struct cache *ca;

		btree_mark_meta(c);

		c->min_prio = initial_prio;

		for_each_cache(ca, c)
			for_each_bucket(b, ca) {
				b->last_gc = b->gc_gen;
				c->need_gc = max(c->need_gc, bucket_gc_gen(b));

				if (b->prio)
					c->min_prio = min(c->min_prio, b->prio);
			}

		for_each_cache(ca, c)
			for_each_bucket(b, ca)
				if (!atomic_read(&b->pin))
					bucket_add_heap(ca, b);
	}

	const char *err = "cannot allocate memory";
	struct cached_dev *d, *t;
	struct cache *ca;
	struct search s;
	search_init_stack(&s);

	for_each_cache(ca, c)
		c->rescale_value += bucket_to_sector(c, ca->sb.nbuckets) / 2048;
	set_gc_sectors(c);

	if (CACHE_SYNC(&c->sb)) {
		LIST_HEAD(journal);
		struct bkey *k;
		struct jset *j;

		err = "cannot allocate memory for journal";
		if (btree_journal_read(c, &journal, &s))
			goto err;

		err = "no journal entries found";
		if (list_empty(&journal))
			goto err;

		j = &list_entry(journal.prev, struct journal_replay, list)->j;

		err = "IO error reading priorities";
		for_each_cache(ca, c) {
			ca->prio_start = j->prio_bucket[ca->sb.nr_this_dev];
			if (prio_read(ca, ca->prio_start))
				goto err;
		}

		k = &j->btree_root;

		err = "bad btree root";
		if (__ptr_invalid(c, j->btree_level + 1, k))
			goto err;

		err = "error reading btree root";
		c->root = get_bucket(c, k, j->btree_level, true, &s.insert);
		if (IS_ERR_OR_NULL(c->root))
			goto err;

		list_del_init(&c->root->lru);
		rw_unlock(true, c->root);

		k = &j->uuid_bucket;

		err = "bad uuid pointer";
		if (__ptr_invalid(c, 1, k))
			goto err;

		bkey_copy(&c->uuid_bucket, k);
		uuid_io(c, READ_SYNC, k, &s.insert);

		err = "error in recovery";
		if (btree_root(check, c, &s))
			goto err;

		btree_journal_mark(c, &journal);

		fill_heap();
		c->journal.seq = j->seq;

		c->journal.cur = c->journal.w;
		btree_journal_next(c);
		btree_journal_replay(c, &journal, &s);
	} else {
		printk(KERN_NOTICE "bcache: invalidating existing data\n");

		for_each_cache(ca, c) {
			ca->sb.keys = clamp_t(int, ca->sb.nbuckets / 100,
					      2, SB_JOURNAL_BUCKETS);

			for (int i = 0; i < ca->sb.keys; i++)
				ca->sb.d[i] = ca->sb.first_bucket + i;

			cache_init_journal(ca);
		}

		fill_heap();

		err = "cannot allocate new UUID bucket";
		if (uuid_write(c))
			goto err;

		err = "cannot allocate new btree root";
		c->root = btree_alloc(c, 0, &s.insert);
		if (IS_ERR_OR_NULL(c->root))
			goto err;

		bkey_copy_key(&c->root->key, &MAX_KEY);
		btree_write(c->root, true, &s);

		for_each_cache(ca, c) {
			free_some_buckets(ca);
			prio_write(ca, &s.insert);
		}

		closure_sync(&s.insert);
		set_new_root(c->root);
		rw_unlock(true, c->root);

		/* first journal entry doesn't get written until after cache is
		 * set to sync */
		SET_CACHE_SYNC(&c->sb, true);

		c->journal.cur = c->journal.w;
		btree_journal_next(c);
		btree_journal_meta(c, &s.insert);
	}

	closure_sync(&s.insert);
	c->sb.last_mount = get_seconds();
	write_super(c, &s.insert);

	list_for_each_entry_safe(d, t, &uncached_devices, list)
		register_dev_on_set(c, d);

	return;
err:
	/* XXX: test this, it's broken */
	cache_set_error(c, err);
}

static bool can_attach_cache(struct cache *c, struct cache_set *s)
{
	return c->sb.block_size == s->sb.block_size &&
		c->sb.bucket_size == s->sb.block_size &&
		c->sb.nr_in_set == s->sb.nr_in_set;
}

static const char *register_cache_set(struct cache *c)
{
	static struct attribute *files[] = {
		&sysfs_unregister,
		&sysfs_synchronous,
		&sysfs_bucket_size,
		&sysfs_block_size,
		&sysfs_tree_depth,
		&sysfs_root_usage_percent,
		&sysfs_btree_avg_keys_written,
		&sysfs_btree_cache_size,

		&sysfs_average_seconds_between_gc,
		&sysfs_gc_ms_max,
		&sysfs_seconds_since_gc,
		&sysfs_btree_nodes,
		&sysfs_btree_used_percent,
		&sysfs_average_key_size,
		&sysfs_dirty_data,

		&sysfs_writeback_keys_done,
		&sysfs_writeback_keys_failed,
		&sysfs_active_journal_entries,
		&sysfs_clear_stats,

		&sysfs_bypassed,
		&sysfs_cache_hits,
		&sysfs_cache_misses,
		&sysfs_cache_hit_ratio,
		&sysfs_cache_readaheads,
		&sysfs_cache_miss_collisions,
		NULL
	};
	static const struct sysfs_ops ops = {
		.show = show_cache_set,
		.store = store_cache_set
	};
	static struct kobj_type set_obj = {
		.release = unregister_cache_set,
		.sysfs_ops = &ops,
		.default_attrs = files
	};

	char buf[12];
	const char *err;
	struct cache_set *s;

	list_for_each_entry(s, &cache_sets, list)
		if (!memcmp(s->sb.set_uuid, c->sb.set_uuid, 16)) {
			err = "duplicate cache set member";
			if (s->cache[c->sb.nr_this_dev])
				goto err;

			err = "cache sb does not match set";
			if (!can_attach_cache(c, s))
				goto err;

			if (!CACHE_SYNC(&c->sb))
				SET_CACHE_SYNC(&s->sb, false);

			goto found;
		}

	err = "cannot allocate memory";
	s = alloc_cache_set(&c->sb);
	if (!s)
		goto err;

	err = "error creating kobject";
	if (kobject_init_and_add(&s->kobj, &set_obj, bcache_kobj,
				 "%pU", s->sb.set_uuid))
		goto err;

	list_add(&s->list, &cache_sets);
found:
	sprintf(buf, "cache%i", c->sb.nr_this_dev);
	if (sysfs_create_link(&c->kobj, &s->kobj, "set") ||
	    sysfs_create_link(&s->kobj, &c->kobj, buf))
		goto err;

	if (c->sb.seq > s->sb.seq) {
		s->sb.version		= c->sb.version;
		memcpy(s->sb.set_uuid, c->sb.set_uuid, 16);
		s->sb.flags		= c->sb.flags;
		s->sb.seq		= c->sb.seq;
		pr_debug("set version = %llu", s->sb.version);
	}

	c->set = s;
	c->set->cache[c->sb.nr_this_dev] = c;
	s->cache_by_alloc[s->caches_loaded++] = c;

	if (s->caches_loaded == s->sb.nr_in_set)
		run_cache_set(s);

	return NULL;
err:
	if (s && s->kobj.state_initialized)
		kobject_put(&s->kobj);
	else if (s)
		free_cache_set(s);

	return err;
}

/* Cache device */

static ssize_t show_cache(struct kobject *kobj, struct attribute *attr,
			  char *buf)
{
	struct cache *c = container_of(kobj, struct cache, kobj);

	sysfs_hprint(bucket_size,	bucket_bytes(c));
	sysfs_hprint(block_size,	block_bytes(c));
	sysfs_print(nbuckets,		"%lli", c->sb.nbuckets);
	sysfs_print(heap_size,		"%zu", c->heap.size);
	sysfs_print(discard, "%i", c->discard);
	sysfs_hprint(written, atomic_long_read(&c->sectors_written) << 9);
	sysfs_hprint(btree_written,
		     atomic_long_read(&c->btree_sectors_written) << 9);
	sysfs_hprint(metadata_written,
		     (atomic_long_read(&c->meta_sectors_written) +
		      atomic_long_read(&c->btree_sectors_written)) << 9);

	if (attr == &sysfs_priority_stats) {
		int cmp(const void *l, const void *r)
		{	return *((uint16_t *) l) - *((uint16_t *) r); }

		size_t n = c->sb.nbuckets, i, zero, btree;
		uint64_t sum = 0;
		uint16_t q[7], *p;

		p = kmalloc(c->sb.nbuckets * sizeof(uint16_t), GFP_KERNEL);
		if (!p)
			return -ENOMEM;

		spin_lock(&c->set->bucket_lock);
		for (i = c->sb.first_bucket; i < n; i++)
			p[i] = c->buckets[i].prio;
		spin_unlock(&c->set->bucket_lock);

		sort(p, n, sizeof(uint16_t), cmp, NULL);

		for (i = 0; i < n && !p[i]; i++)
			;
		zero = i;

		for (i = n; i && p[i - 1] == btree_prio; --i)
			;
		btree = n - i;

		for (i = zero; i < n - btree; i++)
			sum += p[i];

		if (n - btree - zero)
			do_div(sum, n - btree - zero);

		for (i = 0; i < 7; i++)
			q[i] = p[zero + (n - zero - btree) * (i + 1) / 8];

		kfree(p);
		return snprintf(buf, PAGE_SIZE,
				"Zero:	%zu%%\n"
				"Btree:	%zu%%\n"
				"Avg:	%llu\n"
				"Q:	[%u, %u, %u, %u, %u, %u, %u]\n",
				zero * 100 / n, btree * 100 / n, sum,
				q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
	}

	return 0;
}

static ssize_t store_cache(struct kobject *kobj, struct attribute *attr,
			   const char *buffer, size_t size)
{
	struct cache *c = container_of(kobj, struct cache, kobj);

	if (blk_queue_discard(bdev_get_queue(c->bdev)))
		sysfs_atoi(discard, c->discard);

	if (attr == &sysfs_clear_stats) {
		atomic_long_set(&c->sectors_written, 0);
		atomic_long_set(&c->btree_sectors_written, 0);
		atomic_long_set(&c->meta_sectors_written, 0);
	}

	return size;
}

static void free_cache(struct kobject *k)
{
	struct cache *c = container_of(k, struct cache, kobj);
	struct discard *d;
	if (!k)
		return;

	/* XXX: wait if prios are being written */

	if (c->set)
		c->set->cache[c->sb.nr_this_dev] = NULL;

	if (!IS_ERR_OR_NULL(c->debug))
		debugfs_remove(c->debug);

	while (!list_empty(&c->discards)) {
		d = list_first_entry(&c->discards, struct discard, list);
		cancel_work_sync(&d->work);
		list_del(&d->list);
		kfree(d);
	}
	if (c->prio_bio)
		bio_put(c->prio_bio);
	if (c->uuid_bio)
		bio_put(c->uuid_bio);

	free_pages((unsigned long) c->disk_buckets, ilog2(bucket_pages(c)));
	vfree(c->buckets);

	if (c->discard_page)
		put_page(c->discard_page);

	free_heap(&c->heap);
	free_fifo(&c->journal);
	free_fifo(&c->btree_freed);
	free_fifo(&c->free_inc);
	free_fifo(&c->free);

	if (c->sb_bio.bi_inline_vecs[0].bv_page)
		put_page(c->sb_bio.bi_io_vec[0].bv_page);

	if (!IS_ERR_OR_NULL(c->bdev))
		close_bdev_exclusive(c->bdev, FMODE_READ|FMODE_WRITE);

	module_put(THIS_MODULE);
	kfree(c);
}

static int register_cache_kobj(struct cache *c)
{
	static struct attribute *files[] = {
		&sysfs_bucket_size,
		&sysfs_block_size,
		&sysfs_nbuckets,
		&sysfs_priority_stats,
		&sysfs_heap_size,
		&sysfs_discard,
		&sysfs_written,
		&sysfs_btree_written,
		&sysfs_metadata_written,
		&sysfs_clear_stats,
		NULL
	};
	static const struct sysfs_ops ops = {
		.show = show_cache,
		.store = store_cache
	};
	static struct kobj_type cache_obj = {
		.release = free_cache,
		.sysfs_ops = &ops,
		.default_attrs = files
	};

	struct kobject *p = &disk_to_dev(c->bdev->bd_disk)->kobj;
	return kobject_init_and_add(&c->kobj, &cache_obj, p, "bcache");
}

static void cache_init_journal(struct cache *c)
{
	if (!c->sb.keys)
		return;

	c->journal_start = c->sb.bucket_size * c->sb.d[0];
	c->journal_end   = c->sb.bucket_size * (c->sb.d[0] + c->sb.keys);

	c->journal_area_start = c->journal_start;
	c->journal_area_end = c->journal_end;
}

static struct cache *alloc_cache(struct cache_sb *sb)
{
	unsigned free;
	struct bucket *b;
	struct cache *c = kzalloc(sizeof(struct cache), GFP_KERNEL);
	if (!c)
		return NULL;

	memcpy(&c->sb, sb, sizeof(struct cache_sb));

	INIT_LIST_HEAD(&c->discards);

	bio_init(&c->sb_bio);
	c->sb_bio.bi_max_vecs	= 1;
	c->sb_bio.bi_io_vec	= c->sb_bio.bi_inline_vecs;

	bio_init(&c->journal_bio);
	c->journal_bio.bi_max_vecs = 8;
	c->journal_bio.bi_io_vec = c->journal_bio.bi_inline_vecs;

	free = max_t(unsigned, c->sb.nbuckets >> 11, 16);
	free = max_t(unsigned, prio_buckets(c) + 4, free);

	if (!init_fifo(&c->btree_freed,	free, GFP_KERNEL) ||
	    !init_fifo(&c->free,	free, GFP_KERNEL) ||
	    !init_fifo(&c->free_inc,	free << 2, GFP_KERNEL) ||
	    !init_fifo(&c->journal,	1023, GFP_KERNEL) ||
	    !init_heap(&c->heap,	c->sb.nbuckets, GFP_KERNEL) ||
	    !(c->discard_page	= alloc_page(__GFP_ZERO|GFP_KERNEL)) ||
	    !(c->buckets	= vmalloc(sizeof(struct bucket) *
					  c->sb.nbuckets)) ||
	    !(c->prio_buckets	= kzalloc(sizeof(uint64_t) * prio_buckets(c) *
					  2, GFP_KERNEL)) ||
	    !(c->disk_buckets	= alloc_bucket_pages(GFP_KERNEL, c)) ||
	    !(c->uuid_bio	= bio_kmalloc(GFP_KERNEL, bucket_pages(c))) ||
	    !(c->prio_bio	= bio_kmalloc(GFP_KERNEL, bucket_pages(c))))
		goto err;

	c->prio_next = c->prio_buckets + prio_buckets(c);

	memset(c->buckets, 0, c->sb.nbuckets * sizeof(struct bucket));
	for_each_bucket(b, c) {
		atomic_set(&b->pin, 0);
		b->heap = -1;
	}

	for (int i = 0; i < 8; i++) {
		struct discard *d = kzalloc(sizeof(*d), GFP_KERNEL);
		if (!d)
			goto err;

		d->c = c;
		INIT_WORK(&d->work, discard_work);
		list_add(&d->list, &c->discards);
	}

	cache_init_journal(c);

	return c;
err:
	__module_get(THIS_MODULE);
	free_cache(&c->kobj);
	return NULL;
}

static const char *register_cache(struct cache_sb *sb, struct page *sb_page,
				  struct block_device *bdev)
{
	char name[BDEVNAME_SIZE];
	const char *err = "cannot allocate memory";
	struct cache *c = alloc_cache(sb);
	if (!c)
		return err;

	c->sb_bio.bi_io_vec[0].bv_page = sb_page;
	c->bdev = bdev;
	c->bdev->bd_holder = c;

	err = "error creating kobject";
	if (register_cache_kobj(c))
		goto err;

	err = register_cache_set(c);
	if (err) {
		kobject_put(&c->kobj);
		goto err_nofree;
	}

#ifdef CONFIG_DEBUG_FS
	if (!IS_ERR_OR_NULL(debug)) {
		char b[BDEVNAME_SIZE];
		bdevname(c->bdev, b);

		c->debug = debugfs_create_file(b, 0400, debug, c,
					       &cache_debug_ops);
	}
#endif
	printk(KERN_DEBUG "bcache: registered cache device %s\n",
	       bdevname(bdev, name));

	if (0) {
err:
		free_cache(&c->kobj);
err_nofree:
		printk(KERN_DEBUG "bcache: error opening %s: %s\n",
		       bdevname(bdev, name), err);
	}
	return NULL;
}

/* Global interfaces/init */

static ssize_t register_bcache(const char *buffer, size_t size)
{
	ssize_t ret = size;
	const char *err = "cannot allocate memory";
	char *path = NULL;
	struct page *sb_page = NULL;
	struct block_device *bdev;
	struct cache_sb *sb;

	if (!try_module_get(THIS_MODULE))
		return -EBUSY;

	sb = kmalloc(sizeof(struct cache_sb), GFP_KERNEL);
	if (!sb)
		goto err;

	mutex_lock(&register_lock);

	path = kstrndup(buffer, size, GFP_KERNEL);
	if (!path)
		goto err;

	err = "failed to open device";
	bdev = open_bdev_exclusive(strim(path), FMODE_READ|FMODE_WRITE, sb);
	if (bdev == ERR_PTR(-EBUSY))
		err = "device busy";

	if (IS_ERR(bdev))
		goto err;

	if (set_blocksize(bdev, 4096))
		goto err;

	err = read_super(sb, bdev, &sb_page);
	if (err)
		goto err_close;

	if (sb->version == CACHE_BACKING_DEV)
		err = register_bdev(sb, sb_page, bdev);
	else
		err = register_cache(sb, sb_page, bdev);

	if (err) {
		put_page(sb_page);
err_close:
		close_bdev_exclusive(bdev, FMODE_READ|FMODE_WRITE);
err:
		module_put(THIS_MODULE);
		printk(KERN_DEBUG "bcache: error opening %s: %s\n", path, err);
		ret = -EINVAL;
	}

	mutex_unlock(&register_lock);
	kfree(path);
	kfree(sb);
	return ret;
}

static ssize_t show(struct kobject *kobj, struct kobj_attribute *attr,
		    char *buf)
{
	sysfs_print(latency_warn_ms, "%i", latency_warn_ms);

	return 0;
}

static ssize_t store(struct kobject *kobj, struct kobj_attribute *attr,
		     const char *buffer, size_t size)
{
	if (attr == &sysfs_register)
		return register_bcache(buffer, size);
	sysfs_atoi(latency_warn_ms, latency_warn_ms);

	return size;
}

static int __init bcache_init(void)
{
	static const struct attribute *files[] = {
		&sysfs_register.attr,
#ifdef DEBUG_LATENCY
		&sysfs_latency_warn_ms.attr,
#endif
		NULL};

	mutex_init(&register_lock);

	bcache_major = register_blkdev(0, "bcache");
	if (bcache_major < 0)
		return bcache_major;

	if (!(search_cache = KMEM_CACHE(search, 0)) ||
	    !(dirty_cache = KMEM_CACHE(dirty, 0)) ||
	    !(delayed = create_workqueue("bcache")) ||
	    !(writeback = create_singlethread_workqueue("bcache_writeback")) ||
	    !(bcache_kobj = kobject_create_and_add("bcache", fs_kobj)) ||
	    sysfs_create_files(bcache_kobj, files))
		goto err;

	debug = debugfs_create_dir("bcache", NULL);

	return 0;
err:
	if (writeback)
		destroy_workqueue(writeback);
	if (delayed)
		destroy_workqueue(delayed);
	if (dirty_cache)
		kmem_cache_destroy(dirty_cache);
	if (search_cache)
		kmem_cache_destroy(search_cache);
	return -ENOMEM;
}

static void bcache_exit(void)
{
	if (!IS_ERR_OR_NULL(debug))
		debugfs_remove_recursive(debug);

	kobject_put(bcache_kobj);
	destroy_workqueue(writeback);
	destroy_workqueue(delayed);
	kmem_cache_destroy(dirty_cache);
	kmem_cache_destroy(search_cache);
	unregister_blkdev(bcache_major, "bcache");
}

module_init(bcache_init);
module_exit(bcache_exit);
