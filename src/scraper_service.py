import asyncio
import logging
import argparse
from datetime import timedelta
from src.scraper import scrape_top_n_movies

logger = logging.getLogger(__name__)

async def periodic_scraper(
    n: int = 500,
    interval_seconds: int = 3600,
    append: bool = True,
    include_tv: bool = False,
    max_per_run: int = None,
):
    \"\"\"Run scraper periodically every `interval_seconds` seconds.
    
    Non-blocking async loop. Can be run as service.
    \"\"\"
    logger.info("Starting periodic scraper: n=%d interval=%ds append=%s", n, interval_seconds, append)
    try:
        while True:
            try:
                logger.info("Scraping top %d movies...", n)
                scrape_top_n_movies(
                    n=n,
                    append=append,
                    include_tv=include_tv,
                    max_per_run=max_per_run,
                )
                logger.info("Scrape complete. Sleeping %ds...", interval_seconds)
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Periodic scraper stopped by user")
                break
            except Exception as exc:
                logger.error("Scrape failed: %s", exc)
                await asyncio.sleep(300)  # backoff
    except asyncio.CancelledError:
        logger.info("Scraper service cancelled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodic TMDB movie scraper service")
    parser.add_argument("--n", type=int, default=500, help="Total movies to maintain")
    parser.add_argument("--interval", type=int, default=3600, help="Scrape interval (seconds)")
    parser.add_argument("--no-append", action="store_true", help="Overwrite CSV instead of append")
    parser.add_argument("--tv", action="store_true", help="Include TV shows/series")
    parser.add_argument("--max-per-run", type=int, help="Max movies per scrape call")
    args = parser.parse_args()

    async def main():
        await periodic_scraper(
            n=args.n,
            interval_seconds=args.interval,
            append=not args.no_append,
            include_tv=args.tv,
            max_per_run=args.max_per_run,
        )

    asyncio.run(main())

