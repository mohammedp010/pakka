/**
 * Phase 8: makes the three Google Forms (A, B, C) that collect human-written shopping questions.
 *
 * Run it in your own Google account: paste this whole file into a new project at script.google.com,
 * pick makeForms, press Run and allow access. The share and edit links appear in the execution log.
 *
 * Each form shows 10 of the 30 FAQs as everyday situations (never the FAQ question itself, so people
 * don't copy its wording), 3 "twists" that should need a different answer, and one optional free question.
 * Hinglish comes first, then English. All but the free question are required. No email or name is
 * collected; a required checkbox records consent to release the answers as CC BY 4.0.
 *
 * Already made the forms? Paste their edit links into EDIT_URLS and run updateForms instead: it rewrites
 * their text in place, so the share links and the responses so far stay.
 *
 * scripts/ingest_human.py reads the FORM block below and maps the CSV columns back by question title,
 * so change titles here and run updateForms, never edit them in the Forms editor.
 */

const EDIT_URLS = {A: "", B: "", C: ""};

const FORM = // form:start
{
  "title": "Shop ko kaise message karoge? / How would you text a shop?",
  "intro": "Main ek open-source project bana raha hoon jo check karta hai ki shopping chatbots Hinglish aur Hindi sahi samajhte hain ya nahi. 10 chhoti situations, ~10 min. Bas waise likho jaise WhatsApp pe likhte ho.\n\nI'm building an open-source project that checks whether shopping chatbots understand Hinglish and Hindi. 10 short situations, about 10 minutes. Just write it the way you'd text on WhatsApp.",
  "thanks": "Shukriya! Kisi dost ko bhi bhej do jo Hinglish mein likhta hai.\nThank you! Please pass it on to a friend who texts in Hinglish.",
  "consent": {
    "title": "Aapke answers public honge / Your answers will be public",
    "help": "Hum aapka naam, email, phone number ya koi bhi personal detail collect nahi karte, sirf aapke answers. Yeh ek open dataset mein CC BY 4.0 licence ke saath public honge. Answers mein bhi apna number, order ID ya address mat likhna.\n\nWe don't collect your name, email, phone number or any other personal details, only your answers. They'll be published in an open dataset under CC BY 4.0. Please don't put your number, order ID or address in your answers either.",
    "choice": "Theek hai, mere answers public ho sakte hain (CC BY 4.0) / I agree, my answers can be public (CC BY 4.0)"
  },
  "parts": [
    {"title": "Part 1/3: apne words mein / in your own words",
     "help": "Har situation pe shop ke support ko kya message bhejoge? Typos chalega. Do tarike se bologe to alag line mein.\nWhat would you text the shop's support? Typos are fine. Two ways? Put each on its own line."},
    {"title": "Part 2/3: bas 3 aur / just 3 more",
     "help": "Same tarah, jaise WhatsApp pe likhoge.\nSame as before: write it the way you'd text."},
    {"title": "Part 3/3: aapka apna sawaal / your own question",
     "help": "Optional."}
  ],
  "other": {"faq": null, "kind": "uncovered",
            "title": "Kabhi kisi shopping app ke support se kuch poocha hai? Woh likho.",
            "help": "Ever asked a shopping app's support something? Write it here."},
  "variants": {
    "A": {
      "situations": [
        {"faq": "faq_01", "kind": "paraphrase", "title": "Aapne kuch return kiya hai, ab jaanna hai paise kitne din mein wapas aayenge.", "help": "You sent back something you bought, and you want to know how many days the money takes to come back."},
        {"faq": "faq_04", "kind": "paraphrase", "title": "Kharidne se pehle jaanna hai ki kaunsi cheezein shop wapas nahi leti.", "help": "Before buying, you want to know if there are things the shop won't take back."},
        {"faq": "faq_07", "kind": "paraphrase", "title": "Order karne wale ho, jaanna hai aam taur pe kitne din mein pahunchta hai.", "help": "You're about to place an order and want to know how soon it usually reaches you."},
        {"faq": "faq_10", "kind": "paraphrase", "title": "Jaanna hai is shop pe kis-kis tarike se payment kar sakte ho.", "help": "You want to know all the different ways you can pay on this shop."},
        {"faq": "faq_13", "kind": "paraphrase", "title": "Dost ne discount code diya hai, par samajh nahi aa raha kahan daalna hai.", "help": "A friend gave you a discount code, and you can't find where to type it."},
        {"faq": "faq_16", "kind": "paraphrase", "title": "Aaj parcel aaya aur andar ka saaman toota hua nikla.", "help": "Your parcel came today, and the thing inside is broken."},
        {"faq": "faq_19", "kind": "paraphrase", "title": "Business ke liye kharid rahe ho, accounts ke liye GST wala bill chahiye.", "help": "You're buying for your business and need a bill with GST details for your accounts."},
        {"faq": "faq_22", "kind": "paraphrase", "title": "Shopping account ka password bhool gaye, login nahi ho raha.", "help": "You forgot the password of your shopping account and can't log in."},
        {"faq": "faq_25", "kind": "paraphrase", "title": "Mixer grinder le rahe ho, soch rahe ho kuch mahine baad kharab ho gaya to kya hoga.", "help": "You're buying a mixer grinder and wonder what happens if it stops working after a few months."},
        {"faq": "faq_28", "kind": "paraphrase", "title": "Bas ek cheez jaldi leni hai, bina sign up kiye.", "help": "You want to buy just one thing quickly, without signing up."}
      ],
      "twists": [
        {"faq": "faq_07", "kind": "number", "title": "Order Sunday tak hi chahiye, kyunki Monday ko aap bahar ja rahe ho.", "help": "You need the order by Sunday because you're travelling on Monday."},
        {"faq": "faq_16", "kind": "negation", "title": "Parcel aaya, saaman bilkul theek hai, bas pasand nahi aaya.", "help": "Your parcel came and nothing is wrong with it, you just don't like it."},
        {"faq": "faq_25", "kind": "entity", "title": "Ek jacket li thi, ek mahine mein uski zip toot gayi.", "help": "You bought a jacket and its zip broke within a month."}
      ]
    },
    "B": {
      "situations": [
        {"faq": "faq_02", "kind": "paraphrase", "title": "App bol raha hai paise hafte bhar pehle wapas bhej diye, par bank mein abhi tak nahi aaye.", "help": "The app says your money was sent back more than a week ago, but it still isn't in your bank account."},
        {"faq": "faq_05", "kind": "paraphrase", "title": "Online pay nahi karna, parcel ghar aaye tab paise dene hain.", "help": "You don't want to pay online; you'd rather pay when the parcel reaches your door."},
        {"faq": "faq_08", "kind": "paraphrase", "title": "Ghante bhar pehle order kiya tha, ab mann badal gaya, abhi ship nahi hua.", "help": "You ordered something an hour ago and changed your mind before it ships."},
        {"faq": "faq_11", "kind": "paraphrase", "title": "Mehenga phone lena hai aur paise har mahine thode thode dene hain.", "help": "You want to buy an expensive phone and pay for it in monthly parts."},
        {"faq": "faq_14", "kind": "paraphrase", "title": "Checkout pe discount code daala, par baar baar reject ho raha hai.", "help": "You typed a discount code at checkout, but it keeps getting rejected."},
        {"faq": "faq_17", "kind": "paraphrase", "title": "Jaanna hai shipping ke alag se paise lagte hain ya nahi.", "help": "You want to know whether you have to pay extra for shipping."},
        {"faq": "faq_20", "kind": "paraphrase", "title": "App mein baar baar 'Plus' dikhta hai, jaanna hai isme kya milta hai aur kitne ka hai.", "help": "You keep seeing 'Plus' in the app and want to know what you get and what it costs."},
        {"faq": "faq_23", "kind": "paraphrase", "title": "Ab yeh shop use nahi karni, account poora hatwana hai.", "help": "You don't want to use this shop any more and want your account removed completely."},
        {"faq": "faq_26", "kind": "paraphrase", "title": "Sofa order kar rahe ho, chahte ho us din aaye jab aap ghar pe ho.", "help": "You're ordering a sofa and want it to arrive on a day you're at home."},
        {"faq": "faq_29", "kind": "paraphrase", "title": "Aap handmade candles banate ho aur unhe is shop pe bechna chahte ho.", "help": "You make handmade candles and want to sell them through this shop."}
      ],
      "twists": [
        {"faq": "faq_17", "kind": "number", "title": "Cart mein sirf ₹150 ka saaman hai, itne chhote order pe shipping ka kya?", "help": "Your cart is only ₹150, and you wonder about shipping on such a small order."},
        {"faq": "faq_08", "kind": "negation", "title": "Galti se order cancel ho gaya, ab wapas chahiye.", "help": "You cancelled an order by mistake and want it back."},
        {"faq": "faq_29", "kind": "entity", "title": "Aap Dubai mein rehte ho aur wahan se is shop pe bechna chahte ho.", "help": "You live in Dubai and want to sell on this shop from there."}
      ]
    },
    "C": {
      "situations": [
        {"faq": "faq_03", "kind": "paraphrase", "title": "Kuch kharida tha jo ab nahi chahiye, use wapas bhejna hai.", "help": "You bought something you don't want any more and want to send it back."},
        {"faq": "faq_06", "kind": "paraphrase", "title": "Order kar diya hai, ab dekhna hai abhi kahan tak pahuncha.", "help": "You placed an order and want to see where it is right now."},
        {"faq": "faq_09", "kind": "paraphrase", "title": "Order ghar ke address pe kiya tha, ab office pe mangwana hai.", "help": "You ordered to your home address, but now you want it delivered to your office."},
        {"faq": "faq_12", "kind": "paraphrase", "title": "Payment fail dikha, par account se paise kat gaye.", "help": "Your payment showed 'failed', but the money was still cut from your account."},
        {"faq": "faq_15", "kind": "paraphrase", "title": "Jo shoes aaye woh ek size chhote hain, ek size bada chahiye.", "help": "The shoes you got are one size too small, and you want the next size up."},
        {"faq": "faq_18", "kind": "paraphrase", "title": "Aap chhote town mein rehte ho, pata nahi shop wahan deliver karti hai ya nahi.", "help": "You live in a small town and aren't sure the shop delivers to your area."},
        {"faq": "faq_21", "kind": "paraphrase", "title": "Shop ki Plus membership li thi, ab band karni hai.", "help": "You joined the shop's Plus membership and now want to stop it."},
        {"faq": "faq_24", "kind": "paraphrase", "title": "Shop mein kisi insaan se baat karni hai, chat ya phone pe.", "help": "You want to talk to a real person at the shop, by chat or phone."},
        {"faq": "faq_27", "kind": "paraphrase", "title": "Kisi ne is shop ka gift card diya hai, samajh nahi aa raha kaise use karein.", "help": "Someone gave you a gift card for this shop, and you're not sure how to use it."},
        {"faq": "faq_30", "kind": "paraphrase", "title": "App next time ke liye card save karne ko bol raha hai, darr hai safe hai ya nahi.", "help": "The app offers to remember your debit card for next time, and you're worried it isn't safe."}
      ],
      "twists": [
        {"faq": "faq_03", "kind": "number", "title": "Saaman mile 20 din ho gaye, ab wapas karna hai.", "help": "You got the item 20 days ago and now want to send it back."},
        {"faq": "faq_30", "kind": "negation", "title": "Pehle card save kar diya tha, ab hatana hai.", "help": "You saved your card earlier and now want it removed."},
        {"faq": "faq_18", "kind": "entity", "title": "Bhai Nepal mein rehta hai, use is shop se gift bhejna hai.", "help": "Your brother lives in Nepal and you want to send him a gift from this shop."}
      ]
    }
  }
}
// form:end
;

function makeForms() {
  for (const [name, variant] of Object.entries(FORM.variants)) {
    const form = FormApp.create(`${FORM.title} (form ${name})`)
      .setDescription(FORM.intro)
      .setConfirmationMessage(FORM.thanks)
      .setCollectEmail(false)
      .setAllowResponseEdits(false)
      .setShowLinkToRespondAgain(false);
    form.addCheckboxItem().setTitle(FORM.consent.title).setHelpText(FORM.consent.help)
      .setChoiceValues([FORM.consent.choice]).setRequired(true);
    for (const q of layout(variant)) {
      if (FORM.parts.includes(q)) form.addSectionHeaderItem().setTitle(q.title).setHelpText(q.help);
      else form.addParagraphTextItem().setTitle(q.title).setHelpText(q.help).setRequired(q !== FORM.other);
    }
    try {
      form.setPublished(true);  // newer Forms start unpublished; older FormApp has no such method and needs no step
    } catch (e) {}
    Logger.log(`Form ${name}\n  share: ${form.getPublishedUrl()}\n  edit:  ${form.getEditUrl()}`);
  }
}

// Every item after the consent box, in form order: each part's header, then its questions.
function layout(variant) {
  const groups = [variant.situations, variant.twists, [FORM.other]];
  return FORM.parts.flatMap((part, i) => [part, ...groups[i]]);
}

function updateForms() {
  for (const [name, url] of Object.entries(EDIT_URLS)) {
    if (!url) throw new Error(`Paste form ${name}'s edit link into EDIT_URLS first`);
    const form = FormApp.openByUrl(url).setTitle(`${FORM.title} (form ${name})`)
      .setDescription(FORM.intro).setConfirmationMessage(FORM.thanks);
    const items = form.getItems(), want = layout(FORM.variants[name]);
    if (items.length !== want.length + 1) throw new Error(`Form ${name} has ${items.length} items, expected ${want.length + 1}`);
    want.forEach((q, i) => items[i + 1].setTitle(q.title).setHelpText(q.help));  // items[0] is the consent box, left as is
    Logger.log(`Form ${name} updated`);
  }
}
