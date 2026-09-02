"""Alternative stage 03: stable rewrite factors from the complete profile."""

from __future__ import annotations

from common.profile_factor_bank import build


def main() -> None:
    from common.runtime import config_parser, stage_path, teacher_client
    from common.utils import load_config

    args = config_parser("03 - Build full-profile residual rewrite factors").parse_args()
    config = load_config(args.config)
    build(
        stage_path(config, "retrieve"),
        stage_path(config, "factors"),
        config,
        teacher_client(config),
    )


if __name__ == "__main__":
    main()
