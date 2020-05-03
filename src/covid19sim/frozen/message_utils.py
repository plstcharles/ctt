import dataclasses
import numpy as np
import typing

message_uid_bit_count = 4
message_uid_mask = np.uint8((1 << message_uid_bit_count) - 1)
UIDType = np.uint8

risk_level_bit_count = 4
risk_level_mask = np.uint8((1 << risk_level_bit_count) - 1)
RiskLevelType = np.uint8

# GENERAL NOTES ON THE MESSAGE DATA SPEC AS OF 2020/05/01:
# - in an update message, the transmitted UID is the UID at the time of the encounter
# - the encounter timestamp in both encounter/update messages is discretized at the day level
# - the update timestamp in update messages is discretized at the day level and rotating (1-31)
# - the new risk level in update messages is always chained to previously updated risk levels

# TODO: change naive code below to support the rotating day-of-month timestamp in updates


def get_days_offset(
        time_old: np.uint64,
        time_new: np.uint64,
        ticks_per_day: np.uint64 = 24 * 60 * 60,  # assumes one tick per second
) -> float:
    """Returns the number of days between two timestamps."""
    tick_diff = get_ticks_offset(time_old, time_new)
    return tick_diff / ticks_per_day


def get_ticks_offset(
        time_old: np.uint64,
        time_new: np.uint64
) -> int:
    """Returns the number of ticks between two timestmaps."""
    return int(time_new) - int(time_old)


def create_new_uid(rng=None) -> np.uint8:
    """Returns a randomly initialized uid."""
    if rng is None:
        rng = np.random
    return np.uint8(rng.randint(0, 1 << message_uid_bit_count))


def update_uid(uid: np.uint8, rng=None) -> np.uint8:
    """Updates a provided uid by left-shifting it and adding a random bit."""
    if rng is None:
        rng = np.random
    assert 0 <= uid <= message_uid_mask
    return np.uint8(((uid << 1) + rng.randint(0, 2)) & message_uid_mask)


@dataclasses.dataclass
class EncounterMessage:
    """Contains all the observed+unobserved data related to a user encounter message."""

    #####################
    # observed variables

    uid: np.uint8
    """Unique Identifier (UID) of the encountered user."""

    risk_level: np.uint8
    """Quantified risk level of the encountered user."""

    encounter_time: np.uint64
    """Discretized encounter timestamp."""

    #############################################
    # unobserved variables (for debugging only!)

    _sender_uid: typing.Optional[np.uint64] = None
    """Real Unique Identifier (UID) of the encountered user."""

    _receiver_uid: typing.Optional[np.uint64] = None
    """Real Unique Identifier (UID) of the user receiving the message."""

    _real_encounter_time: typing.Optional[np.uint64] = None
    """Real encounter timestamp."""

    _exposition_event: typing.Optional[bool] = None
    """Flags whether this encounter corresponds to an exposition event for the receiver."""


@dataclasses.dataclass
class UpdateMessage:
    """Contains all the observed+unobserved data related to a user update message."""

    #####################
    # observed variables

    uid: np.uint8
    """Unique Identifier (UID) of the updater at the time of the encounter."""

    old_risk_level: np.uint8
    """Previous quantified risk level of the updater."""

    new_risk_level: np.uint8
    """New quantified risk level of the updater."""

    encounter_time: np.uint64
    """Discretized encounter timestamp."""

    update_time: np.uint64
    """Update generation timestamp."""  # TODO: this might be a 1-31 rotating day id?

    #############################################
    # unobserved variables (for debugging only!)

    _sender_uid: typing.Optional[np.uint64] = None
    """Real Unique Identifier (UID) of the updater."""

    _receiver_uid: typing.Optional[np.uint64] = None
    """Real Unique Identifier (UID) of the user receiving the message."""

    _real_encounter_time: typing.Optional[np.uint64] = None
    """Real encounter timestamp."""

    _real_update_time: typing.Optional[np.uint64] = None
    """Real update generation timestamp."""

    _update_reason: typing.Optional[str] = None
    """Reason why this update message was sent (for debugging)."""


def create_update_message(
        encounter_message: EncounterMessage,
        new_risk_level: np.uint8,
        current_time: np.uint64,
        update_reason: typing.Optional[str] = None,
) -> UpdateMessage:
    """Creates and returns an update message for a given encounter.

    Args:
        encounter_message: the encounter message for which to create an update.
        new_risk_level: the new risk level of the sender.
        current_time: the current time of the simulation.
        update_reason: the (optional) reason why this update is being generated.

    Returns:
        The update message object to send.
    """
    assert new_risk_level <= message_uid_mask
    assert current_time >= encounter_message.encounter_time
    return UpdateMessage(
        uid=encounter_message.uid,
        old_risk_level=encounter_message.risk_level,
        new_risk_level=new_risk_level,
        encounter_time=encounter_message.encounter_time,
        update_time=current_time,  # TODO: discretize if needed? @@@
        _sender_uid=encounter_message._sender_uid,
        _receiver_uid=encounter_message._receiver_uid,
        _real_encounter_time=encounter_message._real_encounter_time,
        _real_update_time=current_time,
        _update_reason=update_reason,
    )


def create_encounter_from_update_message(
        update_message: UpdateMessage,
) -> EncounterMessage:
    """Creates and returns an encounter message for a given update message.

    It may happen later that an update message is received for which we cannot find a
    corresponding encounter. This will allow us to create a dummy encounter instead.

    Args:
        update_message: the update message.

    Returns:
        The compatible encounter message object, already updated to the latest risk level.
    """
    return EncounterMessage(
        uid=update_message.uid,
        risk_level=update_message.new_risk_level,
        encounter_time=update_message.encounter_time,
        _sender_uid=update_message._sender_uid,
        _receiver_uid=update_message._receiver_uid,
        _real_encounter_time=update_message._real_encounter_time,
        _exposition_event=None,  # cannot properly deduct if the original encounter was an exposition
    )


def create_updated_encounter_with_message(
        encounter_message: EncounterMessage,
        update_message: UpdateMessage,
) -> EncounterMessage:
    """Creates and returns a new encounter message based on the update of the provided one.

    This function will throw if any "observed" parameters of the messages are mismatched, and
    will silently set unobserved values to `None` if those are mismatched.

    Args:
        encounter_message: the encounter message object to create an updated copy from.
        update_message: the update message that will provide the new risk level.

    Returns:
        A newly instantiated encounter message with the updated attributes.
    """
    assert encounter_message.uid == update_message.uid
    assert encounter_message.risk_level == update_message.old_risk_level
    assert encounter_message.encounter_time == update_message.encounter_time
    return EncounterMessage(
        uid=update_message.uid,
        risk_level=update_message.new_risk_level,
        encounter_time=update_message.encounter_time,
        _sender_uid=update_message._sender_uid if
            update_message._sender_uid == encounter_message._sender_uid else None,
        _receiver_uid=update_message._receiver_uid if
            update_message._receiver_uid == encounter_message._receiver_uid else None,
        _real_encounter_time=update_message._real_encounter_time if
            update_message._real_encounter_time == encounter_message._real_encounter_time else None,
        _exposition_event=encounter_message._exposition_event,
    )


def find_encounter_match_score(
        msg_old: EncounterMessage,
        msg_new: EncounterMessage,
        ticks_per_day: np.uint64 = 24 * 60 * 60,  # assumes one tick per second
):
    """Returns a 'match score' between two encounter messages.

    This function will consider both encounter UIDs as well as their
    discretized timestamp to determine how likely it is that both come
    from the same sender.

    A negative return value means the two messages cannot come from
    the same sender. A zero value means that no matching can be
    established with certainty. A positive value indicates a potential
    match. A higher bit count means the match is more likely. If the
    returned value is equal to `message_uid_bit_count`, the match
    is perfect.

    Returns:
        The match score (an integer in `[-1,message_uid_bit_count]`).
    """
    # TODO: determine if round/ceil should be applied...
    day_offset = int(get_days_offset(
        msg_old.encounter_time,
        msg_new.encounter_time,
        ticks_per_day,
    ))
    assert day_offset >= 0 and \
        0 <= msg_new.uid <= message_uid_mask and \
        0 <= msg_old.uid <= message_uid_mask
    if day_offset >= 4:
        return 0
    if day_offset == 0 and msg_old.uid == msg_new.uid:
        return message_uid_bit_count
    old_uid_mask = message_uid_mask << day_offset
    old_uid = ((msg_old.uid << day_offset) & message_uid_mask)
    if (msg_new.uid & old_uid_mask) == old_uid:
        return message_uid_bit_count - day_offset
    return -1


GenericMessageType = typing.Union[EncounterMessage, UpdateMessage]
