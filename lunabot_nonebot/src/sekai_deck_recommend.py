from typing import Optional, Dict, Any, List, Union

class DeckRecommendCardConfig:
    """
    Card config for a specific rarity
    Attributes:
        disable (bool): Disable this rarity, default is False
        level_max (bool): Always use max level, default is False
        episode_read (bool): Always use read episode, default is False
        master_max (bool): Always use max master rank, default is False
        skill_max (bool): Always use max skill level, default is False
        canvas (bool): Always use canvas bonus, default is False
    """
    
    disable: Optional[bool] = None
    level_max: Optional[bool] = None
    episode_read: Optional[bool] = None
    master_max: Optional[bool] = None
    skill_max: Optional[bool] = None
    canvas: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            'disable': self.disable,
            'level_max': self.level_max,
            'episode_read': self.episode_read,
            'master_max': self.master_max,
            'skill_max': self.skill_max,
            'canvas': self.canvas
        }
        return { key: val for key, val in config.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendCardConfig':
        config:DeckRecommendCardConfig = DeckRecommendCardConfig()
        config.disable = data.get('disable')
        config.level_max = data.get('level_max')
        config.episode_read = data.get('episode_read')
        config.master_max = data.get('master_max')
        config.skill_max = data.get('skill_max')
        config.canvas = data.get('canvas')
        return config


class DeckRecommendSingleCardConfig:
    """
    Card config for single card
    Attributes:
        card_id (int): Card ID
        disable (bool): Disable this card, default is False
        level_max (bool): Always use max level, default is False
        episode_read (bool): Always use read episode, default is False
        master_max (bool): Always use max master rank, default is False
        skill_max (bool): Always use max skill level, default is False
        canvas (bool): Always use canvas bonus, default is False
    """
    
    card_id: int = None
    disable: Optional[bool] = None
    level_max: Optional[bool] = None
    episode_read: Optional[bool] = None
    master_max: Optional[bool] = None
    skill_max: Optional[bool] = None
    canvas: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            'card_id': self.card_id,
            'disable': self.disable,
            'level_max': self.level_max,
            'episode_read': self.episode_read,
            'master_max': self.master_max,
            'skill_max': self.skill_max,
            'canvas': self.canvas
        }
        return { key: val for key, val in config.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendSingleCardConfig':
        config:DeckRecommendSingleCardConfig = DeckRecommendSingleCardConfig()
        config.card_id = data.get('card_id')
        config.disable = data.get('disable')
        config.level_max = data.get('level_max')
        config.episode_read = data.get('episode_read')
        config.master_max = data.get('master_max')
        config.skill_max = data.get('skill_max')
        config.canvas = data.get('canvas')
        return config


class DeckRecommendSaOptions:
    """
    Simulated annealing options
    Attributes:
        run_num (int): Number of simulated annealing runs, default is 20
        seed (int): Random seed, leave it None or use -1 for random seed, default is None
        max_iter (int): Maximum iterations, default is 1000000
        max_no_improve_iter (int): Maximum iterations without improvement, default is 10000
        time_limit_ms (int): Time limit of each run in milliseconds, default is 200
        start_temprature (float): Start temperature, default is 1e8
        cooling_rate (float): Cooling rate, default is 0.999
        debug (bool): Whether to print debug information, default is False
    """
    run_num: Optional[int] = None
    seed: Optional[int] = None
    max_iter: Optional[int] = None
    max_no_improve_iter: Optional[int] = None
    time_limit_ms: Optional[int] = None
    start_temprature: Optional[float] =None
    cooling_rate: Optional[float] = None   
    debug: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            'run_num': self.run_num,
            'seed': self.seed,
            'max_iter': self.max_iter,
            'max_no_improve_iter': self.max_no_improve_iter,
            'time_limit_ms': self.time_limit_ms,
            'start_temprature': self.start_temprature,
            'cooling_rate': self.cooling_rate,
            'debug': self.debug,
        }
        return { key: val for key, val in options.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendSaOptions':
        options: DeckRecommendSaOptions = DeckRecommendSaOptions()
        options.run_num = data.get('run_num')
        options.seed = data.get('seed')
        options.max_iter = data.get('max_iter')
        options.max_no_improve_iter = data.get('max_no_improve_iter')
        options.time_limit_ms = data.get('time_limit_ms')
        options.start_temprature = data.get('start_temprature')
        options.cooling_rate = data.get('cooling_rate')
        options.debug = data.get('debug')
        return options


class DeckRecommendGaOptions:
    """
    Genetic algorithm options
    Attributes:
        seed (int): Random seed, leave it None or use -1 for random seed, default is None
        debug (bool): Whether to print debug information, default is False
        max_iter (int): Maximum iterations, default is 1000000
        max_no_improve_iter (int): Maximum iterations without improvement, default is 5
        pop_size (int): Population size, default is 10000
        parent_size (int): Parent size, default is 1000
        elite_size (int): Elite size, default is 0
        crossover_rate (float): Crossover rate, default is 1.0
        base_mutation_rate (float): Base mutation rate, default is 0.1
        no_improve_iter_to_mutation_rate (float): Rate of no improvement iterations to mutation rate (mutation_rate = base_mutation_rate + no_improve_iter * no_improve_iter_to_mutation_rate), default is 0.02
    """
    seed: Optional[int] = None
    debug: Optional[bool] = None
    max_iter: Optional[int] = None
    max_no_improve_iter: Optional[int] = None
    pop_size: Optional[int] = None
    parent_size: Optional[int] = None
    elite_size: Optional[int] = None
    crossover_rate: Optional[float] = None
    base_mutation_rate: Optional[float] = None
    no_improve_iter_to_mutation_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            'seed': self.seed,
            'debug': self.debug,
            'max_iter': self.max_iter,
            'max_no_improve_iter': self.max_no_improve_iter,
            'pop_size': self.pop_size,
            'parent_size': self.parent_size,
            'elite_size': self.elite_size,
            'crossover_rate': self.crossover_rate,
            'base_mutation_rate': self.base_mutation_rate,
            'no_improve_iter_to_mutation_rate': self.no_improve_iter_to_mutation_rate,
        }
        return { key: val for key, val in options.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendGaOptions':
        options: DeckRecommendGaOptions = DeckRecommendGaOptions()
        options.seed = data.get('seed')
        options.debug = data.get('debug')
        options.max_iter = data.get('max_iter')
        options.max_no_improve_iter = data.get('max_no_improve_iter')
        options.pop_size = data.get('pop_size')
        options.parent_size = data.get('parent_size')
        options.elite_size = data.get('elite_size')
        options.crossover_rate = data.get('crossover_rate')
        options.base_mutation_rate = data.get('base_mutation_rate')
        options.no_improve_iter_to_mutation_rate = data.get('no_improve_iter_to_mutation_rate')
        return options


class DeckRecommendOptions:
    """
    Deck recommend options
    Attributes:
        target (str): Target of the recommendation in ["score", "power", "skill", "bonus"], default is "score"
        algorithm (str): "dfs" for brute force, "sa" for simulated annealing, "ga" for genetic algorithm, default is "ga"
        region (str): Region in ["jp", "en", "tw", "kr", "cn"]
        user_data_file_path (str): File path of user suite json
        user_data_str (str | bytes): String or bytes of user suite json
        live_type (str): Live type in ["multi", "solo", "auto", "challenge", "challenge_auto", "mysekai"]
        music_id (int): Music ID
        music_diff (str): Music difficulty in ["easy", "normal", "hard", "expert", "master", "append"]
        event_id (int): Event ID, leave it None to use no-event or unit-attr-specificed recommendation
        event_attr (str): Attribute of unit-attr-specificed recommendation, only available when event_id is None. In ["mysterious", "cute", "cool", "pure", "happy"]
        event_unit (str): Unit of unit-attr-specificed recommendation, only available when event_id is None. In ["light_sound", "idol", "street", "theme_park", "school_refusal", "piapro"]
        event_type (str): Event type of unit-attr-specificed/no-event recommendation, only available when event_id is None. In ["marathon", "cheerful_carnival"]
        world_bloom_character_id (int): World bloom character ID, only required when event is world bloom
        challenge_live_character_id (int): Challenge live character ID, only required when live is challenge live
        limit (int): Limit of returned decks, default is 10. No guarantee to return this number of decks if not enough cards
        member (int): Number of members in the deck, default is 5
        timeout_ms (int): Timeout in milliseconds, default is None
        rarity_1_config (DeckRecommendCardConfig): Card config for rarity 1
        rarity_2_config (DeckRecommendCardConfig): Card config for rarity 2
        rarity_3_config (DeckRecommendCardConfig): Card config for rarity 3
        rarity_birthday_config (DeckRecommendCardConfig): Card config for birthday cards
        rarity_4_config (DeckRecommendCardConfig): Card config for rarity 4
        single_card_configs (List[DeckRecommendSingleCardConfig]): Card config for single cards that will override rarity configs.
        filter_other_unit (bool): Whether to filter out other units for banner event, default is False
        fixed_cards (List[int]): List of card IDs that always included in the deck, default is None
        fixed_characters (List[int]): List of character IDs that always included in the deck (first is always leader), cannot used in challenge live, cannot used with fixed_cards together, default is None
        target_bonus_list (List[int]): List of target event bonus, required when target is "bonus"
        skill_reference_choose_strategy (str): Strategy for bfes skill reference choose in ["average", "max", "min"], default is "average"
        keep_after_training_state (bool): Whether to keep after-training state of bfes cards, default is False
        multi_live_teammate_score_up (int): Score up of single multi-live teammate, default is None (None means copying self score up)
        multi_live_teammate_power (int): Power of single multi-live teammate, default is None (None means copying self power)
        best_skill_as_leader (bool): Whether to use the best skill card as leader, default is True
        multi_live_score_up_lower_bound (float): Lower bound of multi live score up, only available when live_type is "multi", default is 0
        skill_order_choose_strategy (str): Strategy for skill order choose in ["average", "max", "min", "specific"], default is "average"
        specific_skill_order (List[int]): Specific skill order starting from 0, only required when skill_order_choose_strategy is "specific", default is None
        sa_options (DeckRecommendSaOptions): Simulated annealing options
        ga_options (DeckRecommendGaOptions): Genetic algorithm options
    """
    target: Optional[str] = None
    algorithm: Optional[str] = None
    region: str = None
    user_data_file_path: Optional[str] = None
    user_data_str: Optional[Union[str, bytes]] = None
    live_type: str = None
    music_id: int = None
    music_diff: str = None
    event_id: Optional[int] = None
    event_attr: Optional[str] = None
    event_unit: Optional[str] = None
    event_type: Optional[str] = None
    world_bloom_character_id: Optional[int] = None
    challenge_live_character_id: Optional[int] = None
    limit: Optional[int] = None
    member: Optional[int] = None
    timeout_ms: Optional[int] = None
    rarity_1_config: Optional[DeckRecommendCardConfig] = None
    rarity_2_config: Optional[DeckRecommendCardConfig] = None
    rarity_3_config: Optional[DeckRecommendCardConfig] = None
    rarity_birthday_config: Optional[DeckRecommendCardConfig] = None
    rarity_4_config: Optional[DeckRecommendCardConfig] = None
    single_card_configs: Optional[List[DeckRecommendSingleCardConfig]] = None
    filter_other_unit: Optional[bool] = None
    fixed_cards: Optional[List[int]] = None
    fixed_characters: Optional[List[int]] = None
    target_bonus_list: Optional[List[int]] = None
    skill_reference_choose_strategy: Optional[str] = None
    keep_after_training_state: Optional[bool] = None
    multi_live_teammate_score_up: Optional[int] = None
    multi_live_teammate_power: Optional[int] = None
    best_skill_as_leader: Optional[bool] = None
    multi_live_score_up_lower_bound: Optional[float] = None
    skill_order_choose_strategy: Optional[str] = None
    specific_skill_order: Optional[List[int]] = None
    sa_options: Optional[DeckRecommendSaOptions] = None
    ga_options: Optional[DeckRecommendGaOptions] = None
    def __init__(self, options: Optional['DeckRecommendOptions'] = None):
        if options is None:
            return
        self.target = options.target
        self.algorithm = options.algorithm
        self.region = options.region
        self.user_data_file_path = options.user_data_file_path
        self.user_data_str = options.user_data_str
        self.live_type = options.live_type
        self.music_id = options.music_id
        self.music_diff = options.music_diff
        self.event_id = options.event_id
        self.event_attr = options.event_attr
        self.event_unit = options.event_unit
        self.event_type = options.event_type
        self.world_bloom_character_id = options.world_bloom_character_id
        self.challenge_live_character_id = options.challenge_live_character_id
        self.limit = options.limit
        self.member = options.member
        self.timeout_ms = options.timeout_ms
        self.rarity_1_config = options.rarity_1_config
        self.rarity_2_config = options.rarity_2_config
        self.rarity_3_config = options.rarity_3_config
        self.rarity_birthday_config = options.rarity_birthday_config
        self.rarity_4_config = options.rarity_4_config
        self.single_card_configs = options.single_card_configs
        self.filter_other_unit = options.filter_other_unit
        self.fixed_cards = options.fixed_cards
        self.fixed_characters = options.fixed_characters
        self.target_bonus_list = options.target_bonus_list
        self.skill_reference_choose_strategy = options.skill_reference_choose_strategy
        self.keep_after_training_state = options.keep_after_training_state
        self.multi_live_teammate_score_up = options.multi_live_teammate_score_up
        self.multi_live_teammate_power = options.multi_live_teammate_power
        self.best_skill_as_leader = options.best_skill_as_leader
        self.multi_live_score_up_lower_bound = options.multi_live_score_up_lower_bound
        self.skill_order_choose_strategy = options.skill_order_choose_strategy
        self.specific_skill_order = options.specific_skill_order
        self.sa_options = options.sa_options
        self.ga_options = options.ga_options
        pass
    def to_dict(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            'target': self.target,
            'algorithm': self.algorithm,
            'region': self.region,
            'user_data_file_path': self.user_data_file_path,
            'user_data_str': self.user_data_str,
            'live_type': self.live_type,
            'music_id': self.music_id,
            'music_diff': self.music_diff,
            'event_id': self.event_id,
            'event_attr': self.event_attr,
            'event_unit': self.event_unit,
            'event_type': self.event_type,
            'world_bloom_character_id': self.world_bloom_character_id,
            'challenge_live_character_id': self.challenge_live_character_id,
            'limit': self.limit,
            'member': self.member,
            'timeout_ms': self.timeout_ms,
            'rarity_1_config': self.rarity_1_config.to_dict() if self.rarity_1_config is not None else None,
            'rarity_2_config': self.rarity_2_config.to_dict() if self.rarity_2_config is not None else None,
            'rarity_3_config': self.rarity_3_config.to_dict() if self.rarity_3_config is not None else None,
            'rarity_birthday_config': self.rarity_birthday_config.to_dict() \
                                    if self.rarity_birthday_config is not None \
                                    else None,
            'rarity_4_config': self.rarity_4_config.to_dict() if self.rarity_4_config is not None else None,
            'single_card_configs': [config.to_dict() for config in self.single_card_configs] \
                                    if (self.single_card_configs is not None) \
                                        and (len(self.single_card_configs) > 0) \
                                    else None,
            'filter_other_unit': self.filter_other_unit,
            'fixed_cards': self.fixed_cards,
            'fixed_characters': self.fixed_characters,
            'target_bonus_list': self.target_bonus_list,
            'skill_reference_choose_strategy': self.skill_reference_choose_strategy,
            'keep_after_training_state': self.keep_after_training_state,
            'multi_live_teammate_score_up': self.multi_live_teammate_score_up,
            'multi_live_teammate_power': self.multi_live_teammate_power,
            'best_skill_as_leader': self.best_skill_as_leader,
            'multi_live_score_up_lower_bound': self.multi_live_score_up_lower_bound,
            'skill_order_choose_strategy': self.skill_order_choose_strategy,
            'specific_skill_order': self.specific_skill_order,
            'sa_options': self.sa_options.to_dict() \
                        if self.sa_options is not None \
                        else None,
            'ga_options': self.ga_options.to_dict() \
                        if self.ga_options is not None \
                        else None,
        }
        return { key: val for key, val in options.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendOptions':
        options: DeckRecommendOptions = DeckRecommendOptions()
        options.target = data.get('target')
        options.algorithm = data.get('algorithm')
        options.region = data.get('region')
        options.user_data_file_path = data.get('user_data_file_path')
        options.user_data_str = data.get('user_data_str')
        options.live_type = data.get('live_type')
        options.music_id = data.get('music_id')
        options.music_diff = data.get('music_diff')
        options.event_id = data.get('event_id')
        options.event_attr = data.get('event_attr')
        options.event_unit = data.get('event_unit')
        options.event_type = data.get('event_type')
        options.world_bloom_character_id = data.get('world_bloom_character_id')
        options.challenge_live_character_id = data.get('challenge_live_character_id')
        options.limit = data.get('limit')
        options.member = data.get('member')
        options.timeout_ms = data.get('timeout_ms')
        options.rarity_1_config = DeckRecommendCardConfig.from_dict(config) \
                                if (config := data.get('rarity_1_config')) is not None \
                                else None
        options.rarity_2_config = DeckRecommendCardConfig.from_dict(config) \
                                if (config := data.get('rarity_2_config')) is not None \
                                else None
        options.rarity_3_config = DeckRecommendCardConfig.from_dict(config) \
                                if (config := data.get('rarity_3_config')) is not None \
                                else None
        options.rarity_birthday_config = DeckRecommendCardConfig.from_dict(config) \
                                    if (config := data.get('rarity_birthday_config')) is not None \
                                    else None
        options.rarity_4_config = DeckRecommendCardConfig.from_dict(config) \
                                if (config := data.get('rarity_4_config')) is not None \
                                else None
        options.single_card_configs = [ DeckRecommendSingleCardConfig.from_dict(config) for config in configs ] \
                                 if ((configs := data.get('single_card_configs')) is not None) \
                                    and (len(configs) > 0) \
                                 else None
        options.filter_other_unit = data.get('filter_other_unit')
        options.fixed_cards = data.get('fixed_cards')
        options.fixed_characters = data.get('fixed_characters')
        options.target_bonus_list = data.get('target_bonus_list')
        options.skill_reference_choose_strategy = data.get('skill_reference_choose_strategy')
        options.keep_after_training_state = data.get('keep_after_training_state')
        options.multi_live_teammate_score_up = data.get('multi_live_teammate_score_up')
        options.multi_live_teammate_power = data.get('multi_live_teammate_power')
        options.best_skill_as_leader = data.get('best_skill_as_leader')
        options.multi_live_score_up_lower_bound = data.get('multi_live_score_up_lower_bound')
        options.skill_order_choose_strategy = data.get('skill_order_choose_strategy')
        options.specific_skill_order = data.get('skill_order_choose_strategy')
        options.sa_options = DeckRecommendSaOptions.from_dict(config) \
                            if (config := data.get('sa_options')) is not None \
                            else None
        options.ga_options = DeckRecommendGaOptions.from_dict(config) \
                            if (config := data.get('ga_options')) is not None \
                            else None
        return options

class RecommendCard:
    """
    Card recommendation result
    Attributes:
        card_id (int): Card ID
        total_power (int): Total power of the card
        base_power (int): Base power of the card
        event_bonus_rate (float): Event bonus rate of the card
        master_rank (int): Master rank of the card
        level (int): Level of the card
        skill_level (int): Skill level of the card
        skill_score_up (int): Skill score up of the card
        skill_life_recovery (int): Skill life recovery of the card
        episode1_read (bool): Whether episode 1 is read
        episode2_read (bool): Whether episode 2 is read
        after_training (bool): Whether the card is after special training
        default_image (str): Default image of the card in ["original", "special_training"]
        has_canvas_bonus (bool): Whether the card has canvas bonus
    """
    card_id: int = None
    total_power: int = None
    base_power: int = None
    event_bonus_rate: float = None
    master_rank: int = None
    level: int = None
    skill_level: int = None
    skill_score_up: int = None
    skill_life_recovery: int = None
    episode1_read: bool = None
    episode2_read: bool = None
    after_training: bool = None
    default_image: str = None
    has_canvas_bonus: bool = None

    def to_dict(self) -> Dict[str, Any]:
        card: Dict[str, Any] = {
            'card_id': self.card_id,
            'total_power': self.total_power,
            'base_power': self.base_power,
            'event_bonus_rate': self.event_bonus_rate,
            'master_rank': self.master_rank,
            'level': self.level,
            'skill_level': self.skill_level,
            'skill_score_up': self.skill_score_up,
            'skill_life_recovery': self.skill_life_recovery,
            'episode1_read': self.episode1_read,
            'episode2_read': self.episode2_read,
            'after_training': self.after_training,
            'default_image': self.default_image,
            'has_canvas_bonus': self.has_canvas_bonus,
        }
        return { key: val for key, val in card.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendCard':
        card: RecommendCard = RecommendCard()
        card.card_id = data['card_id']
        card.total_power = data['total_power']
        card.base_power = data['base_power']
        card.event_bonus_rate = data['event_bonus_rate']
        card.master_rank = data['master_rank']
        card.level = data['level']
        card.skill_level = data['skill_level']
        card.skill_score_up = data['skill_score_up']
        card.skill_life_recovery = data['skill_life_recovery']
        card.episode1_read = data['episode1_read']
        card.episode2_read = data['episode2_read']
        card.after_training = data['after_training']
        card.default_image = data['default_image']
        card.has_canvas_bonus = data['has_canvas_bonus']
        return card


class RecommendDeck:
    """
    Deck recommendation result
    Attributes:
        score (int): event point or challenge score of the deck
        live_score (int): Live score of the deck
        mysekai_event_point (int): event point of the deck obtained in mysekai
        total_power (int): Total power of the deck
        base_power (int): Base power of the deck
        area_item_bonus_power (int): Area item bonus power of the deck
        character_bonus_power (int): Character bonus power of the deck
        honor_bonus_power (int): Honor bonus power of the deck
        fixture_bonus_power (int): Fixture bonus power of the deck
        gate_bonus_power (int): Gate bonus power of the deck
        event_bonus_rate (float): Event bonus rate of the deck
        support_deck_bonus_rate (float): Support deck bonus rate of the deck
        multi_live_score_up (float): final score up of the deck in multi live
        cards (List[RecommendCard]): List of recommended cards in the deck
    """
    score: int = None
    live_score: int = None
    mysekai_event_point: int = None
    total_power: int = None
    base_power: int = None
    area_item_bonus_power: int = None
    character_bonus_power: int = None
    honor_bonus_power: int = None
    fixture_bonus_power: int = None
    gate_bonus_power: int = None
    event_bonus_rate: float = None
    support_deck_bonus_rate: float = None
    multi_live_score_up: float = None
    cards: List[RecommendCard] = None

    def to_dict(self) -> Dict[str, Any]:
        deck: Dict[str, Any] = {
            'score': self.score,
            'live_score': self.live_score,
            'mysekai_event_point': self.mysekai_event_point,
            'total_power': self.total_power,
            'base_power': self.base_power,
            'area_item_bonus_power': self.area_item_bonus_power,
            'character_bonus_power': self.character_bonus_power,
            'honor_bonus_power': self.honor_bonus_power,
            'fixture_bonus_power': self.fixture_bonus_power,
            'gate_bonus_power': self.gate_bonus_power,
            'event_bonus_rate': self.event_bonus_rate,
            'support_deck_bonus_rate': self.support_deck_bonus_rate,
            'multi_live_score_up': self.multi_live_score_up,
            'cards': [card.to_dict() for card in self.cards],
        }
        return { key: val for key, val in deck.items() if val is not None }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecommendDeck':
        deck: RecommendDeck = RecommendDeck()
        deck.score = data.get('score')
        deck.live_score = data.get('live_score')
        deck.mysekai_event_point = data.get('mysekai_event_point')
        deck.total_power = data.get('total_power')
        deck.base_power = data.get('base_power')
        deck.area_item_bonus_power = data.get('area_item_bonus_power')
        deck.character_bonus_power = data.get('character_bonus_power')
        deck.honor_bonus_power = data.get('honor_bonus_power')
        deck.fixture_bonus_power = data.get('fixture_bonus_power')
        deck.gate_bonus_power = data.get('gate_bonus_power')
        deck.event_bonus_rate = data.get('event_bonus_rate')
        deck.support_deck_bonus_rate = data.get('support_deck_bonus_rate')
        deck.multi_live_score_up = data.get('multi_live_score_up')
        deck.cards = [RecommendCard.from_dict(card) for card in data.get('cards')]
        return deck


class DeckRecommendResult:
    """
    Deck recommendation result
    Attributes:
        decks (List[RecommendDeck]): List of recommended decks
    """
    decks: List[RecommendDeck] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decks': [ deck.to_dict() for deck in self.decks ]
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckRecommendResult':
        result: DeckRecommendResult = DeckRecommendResult()
        result.decks = [ RecommendDeck.from_dict(deck) for deck in data.get('decks') ]
        return result


class SekaiDeckRecommend:
    """
    Class for event or challenge live deck recommendation  

    Example usage:
    ```
    from sekai_deck_recommend import SekaiDeckRecommend, DeckRecommendOptions
   
    sekai_deck_recommend = SekaiDeckRecommend()

    sekai_deck_recommend.update_masterdata("base/dir/of/masterdata", "jp")
    sekai_deck_recommend.update_musicmetas("file/path/of/musicmetas", "jp")

    options = DeckRecommendOptions()
    options.algorithm = "sa"
    options.region = "jp"
    options.user_data_file_path = "user/data/file/path"
    options.live_type = "multi"
    options.music_id = 74
    options.music_diff = "expert"
    options.event_id = 160
    
    result = sekai_deck_recommend.recommend(options)
    ```

    For more details about the options, please refer docstring of `DeckRecommendOptions` class.
    """

    def __init__(self) -> None:
        ...

    def update_masterdata(self, base_dir: str, region: str) -> None:
        """
        Update master data of the specific region from a local directory
        Args:
            base_dir (str): Base directory of master data
            region (str): Region in ["jp", "en", "tw", "kr", "cn"]
        """
        ...

    def update_masterdata_from_strings(self, data: Dict[str, Union[str, bytes]], region: str) -> None:
        """
        Update master data of the specific region from dictionary of string or bytes
        Args:
            data (Dict[str, bytes]): Dictionary of master data jsons in string or bytes
                example: data = {
                    "cards": "...",
                    "events": "...",
                }
            region (str): Region in ["jp", "en", "tw", "kr", "cn"]
        """
        ...

    def update_musicmetas(self, file_path: str, region: str) -> None:
        """
        Update music metas of the specific region from a local file
        Args:
            file_path (str): File path of music metas
            region (str): Region in ["jp", "en", "tw", "kr", "cn"]
        """
        ...

    def update_musicmetas_from_string(self, data: Union[str, bytes], region: str) -> None:
        """
        Update music metas of the specific region from string or bytes
        Args:
            data (bytes): String or bytes of music metas json
            region (str): Region in ["jp", "en", "tw", "kr", "cn"]
        """
        ...

    def recommend(self, options: DeckRecommendOptions) -> DeckRecommendResult:
        """
        Recommend event or challenge live decks
        Args:
            options (DeckRecommendOptions): Options for deck recommendation
        Returns:
            DeckRecommendResult: Recommended decks sorted by score descending
        """
        ...